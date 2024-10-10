if __name__ == '__main__':
    mse_list = []
    for r in range(100):
        n = 1000000
        d = 10
        dataset = np.random.randint(d, size=(n))
        oue = OUE(d, 1)
        oue.group_user_num = n
        for i in range(n):
            oue.operation_perturb(dataset[i])

        oue.operation_aggregate()
        for j in range(d):
            mse = abs(oue.perturbed_count[j] - oue.aggregated_count[j]) ** 2
        mse = mse / d
    mse_list.append(mse)
    print(mean(mse_list))