from stcal.multiprocessing import compute_num_cores


def test_compute_num_cores():
    n_rows = 20
    max_available_cores = 10
    assert compute_num_cores("none", n_rows, max_available_cores) == 1
    assert compute_num_cores("half", n_rows, max_available_cores) == 5
    assert compute_num_cores("3", n_rows, max_available_cores) == 3
    assert compute_num_cores("7", n_rows, max_available_cores) == 7
    assert compute_num_cores("21", n_rows, max_available_cores) == 10
    assert compute_num_cores("quarter", n_rows, max_available_cores) == 2
    assert compute_num_cores("7.5", n_rows, max_available_cores) == 1
    assert compute_num_cores("one", n_rows, max_available_cores) == 1
    assert compute_num_cores("-5", n_rows, max_available_cores) == 1
    assert compute_num_cores("all", n_rows, max_available_cores) == 10
    assert compute_num_cores("3/4", n_rows, max_available_cores) == 1
    n_rows = 9
    assert compute_num_cores("21", n_rows, max_available_cores) == 9
