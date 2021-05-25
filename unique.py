import numpy as np

list_cor = [[4190091.4195999987, 7410226.618699998], 
    [4190033.2124999985, 7410220.0823], 
    [4190033.2124999985, 7410220.0823], 
    [4190035.7005000003, 7410208.670500003], 
    [4190033.2124999985, 7410220.0823], 
    [4190022.768599998, 7410217.844300002]]

# arr, uniq_cnt = np.unique(list_cor, axis=0, return_counts=True)
# uniq_arr = arr[uniq_cnt==1]

uniq_arr = np.unique(list_cor, axis = 0)

print(uniq_arr)