#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <H5Cpp.h>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <unordered_map>

namespace py = pybind11;

struct SparseData {
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<double> data;
    int nrows;
    int ncols;
};

// Fast parallel sparse matrix subset reader with file-level parallelism
SparseData read_sparse_hdf5_subset(
    const std::vector<std::string>& h5_files,
    const std::string& dataset_path,
    const std::vector<int64_t>& row_indices,
    const std::vector<int64_t>& col_indices,
    int num_threads = 0
) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    int nrows = row_indices.size();
    int ncols = col_indices.size();

    // Create row index lookup
    std::unordered_map<int64_t, int> row_map;
    for (int i = 0; i < nrows; i++) {
        row_map[row_indices[i]] = i;
    }

    // Track column offsets across files
    std::vector<int64_t> col_offsets = {0};

    // Determine which columns belong to which file
    std::vector<std::vector<int64_t>> cols_per_file(h5_files.size());
    std::vector<std::vector<int>> output_cols_per_file(h5_files.size());

    for (size_t file_idx = 0; file_idx < h5_files.size(); file_idx++) {
        H5::H5File file(h5_files[file_idx], H5F_ACC_RDONLY);
        H5::Group group = file.openGroup(dataset_path);

        // Read shape
        H5::DataSet shape_ds = group.openDataSet("shape");
        std::vector<int64_t> shape(2);
        shape_ds.read(shape.data(), H5::PredType::NATIVE_INT64);

        int64_t file_ncols = shape[1];
        int64_t start_col = col_offsets.back();
        col_offsets.push_back(start_col + file_ncols);

        // Find which columns we need from this file
        for (int j = 0; j < ncols; j++) {
            int64_t global_col = col_indices[j];
            if (global_col >= start_col && global_col < start_col + file_ncols) {
                cols_per_file[file_idx].push_back(global_col - start_col);
                output_cols_per_file[file_idx].push_back(j);
            }
        }
    }

    // Result storage - one set per file to avoid contention
    std::vector<std::vector<std::vector<std::pair<int, double>>>> rows_per_file(h5_files.size());
    for (size_t i = 0; i < h5_files.size(); i++) {
        rows_per_file[i].resize(nrows);
    }

    // Process each file in parallel (file-level parallelism)
    #pragma omp parallel for schedule(dynamic) if(h5_files.size() > 1)
    for (size_t file_idx = 0; file_idx < h5_files.size(); file_idx++) {
        if (cols_per_file[file_idx].empty()) continue;

        // Each thread opens its own file handle
        H5::H5File file(h5_files[file_idx], H5F_ACC_RDONLY);
        H5::Group group = file.openGroup(dataset_path);
        H5::DataSet indptr_ds = group.openDataSet("indptr");
        H5::DataSet indices_ds = group.openDataSet("indices");
        H5::DataSet data_ds = group.openDataSet("data");

        // Read shape and indptr
        H5::DataSet shape_ds = group.openDataSet("shape");
        std::vector<int64_t> shape(2);
        shape_ds.read(shape.data(), H5::PredType::NATIVE_INT64);
        int64_t file_ncols = shape[1];

        std::vector<int> indptr_full(file_ncols + 1);
        indptr_ds.read(indptr_full.data(), H5::PredType::NATIVE_INT);

        // Process each column in this file
        for (size_t k = 0; k < cols_per_file[file_idx].size(); k++) {
            int64_t local_col = cols_per_file[file_idx][k];
            int output_col = output_cols_per_file[file_idx][k];

            int start = indptr_full[local_col];
            int end = indptr_full[local_col + 1];
            int col_nnz = end - start;

            if (col_nnz == 0) continue;

            // Read column data
            std::vector<int> col_indices_data(col_nnz);
            std::vector<double> col_data(col_nnz);

            hsize_t count[1] = {(hsize_t)col_nnz};
            hsize_t offset[1] = {(hsize_t)start};

            H5::DataSpace file_space_idx = indices_ds.getSpace();
            file_space_idx.selectHyperslab(H5S_SELECT_SET, count, offset);
            H5::DataSpace mem_space(1, count);
            indices_ds.read(col_indices_data.data(), H5::PredType::NATIVE_INT,
                          mem_space, file_space_idx);

            H5::DataSpace file_space_data = data_ds.getSpace();
            file_space_data.selectHyperslab(H5S_SELECT_SET, count, offset);
            data_ds.read(col_data.data(), H5::PredType::NATIVE_DOUBLE,
                        mem_space, file_space_data);

            // Filter for requested rows (no locking needed - separate storage per file)
            for (int i = 0; i < col_nnz; i++) {
                int64_t row = col_indices_data[i];
                auto it = row_map.find(row);
                if (it != row_map.end()) {
                    int out_row = it->second;
                    rows_per_file[file_idx][out_row].push_back({output_col, col_data[i]});
                }
            }
        }
    }

    // Merge results from all files (parallel merge by row)
    std::vector<std::vector<std::pair<int, double>>> rows(nrows);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nrows; i++) {
        // Estimate size
        size_t total_size = 0;
        for (size_t file_idx = 0; file_idx < h5_files.size(); file_idx++) {
            total_size += rows_per_file[file_idx][i].size();
        }
        rows[i].reserve(total_size);

        // Merge from all files
        for (size_t file_idx = 0; file_idx < h5_files.size(); file_idx++) {
            rows[i].insert(rows[i].end(),
                          rows_per_file[file_idx][i].begin(),
                          rows_per_file[file_idx][i].end());
        }

        // Sort by column
        std::sort(rows[i].begin(), rows[i].end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
    }

    // Build CSR format
    std::vector<int> indptr(nrows + 1, 0);
    std::vector<int> indices;
    std::vector<double> data;

    // Estimate total size
    size_t total_nnz = 0;
    for (int i = 0; i < nrows; i++) {
        total_nnz += rows[i].size();
    }
    indices.reserve(total_nnz);
    data.reserve(total_nnz);

    // Sequential assembly
    for (int i = 0; i < nrows; i++) {
        for (const auto& p : rows[i]) {
            indices.push_back(p.first);
            data.push_back(p.second);
        }
        indptr[i + 1] = indptr[i] + rows[i].size();
    }

    return {indptr, indices, data, nrows, ncols};
}

PYBIND11_MODULE(hdf5_sparse_reader, m) {
    py::class_<SparseData>(m, "SparseData")
        .def_readonly("indptr", &SparseData::indptr)
        .def_readonly("indices", &SparseData::indices)
        .def_readonly("data", &SparseData::data)
        .def_readonly("nrows", &SparseData::nrows)
        .def_readonly("ncols", &SparseData::ncols);

    m.def("read_sparse_hdf5_subset", &read_sparse_hdf5_subset,
          py::arg("h5_files"),
          py::arg("dataset_path"),
          py::arg("row_indices"),
          py::arg("col_indices"),
          py::arg("num_threads") = 0,
          "Fast parallel sparse HDF5 subset reader with file-level parallelism");
}
