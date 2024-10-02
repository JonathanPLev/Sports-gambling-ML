def complex_recursive_function(n, step):
    if n <= 0:
        print("Done")
        return
    else:
        print(n)
        if n % 2 == 0:
            complex_recursive_function(n - step, step + 1)
        else:
            complex_recursive_function(n - step, step)

complex_recursive_function(10, 1)