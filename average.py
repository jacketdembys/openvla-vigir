import numpy as np
import pandas as pd


if __name__ == "__main__":

    # Load from 0
    np.set_printoptions(suppress=True)
    res_zero = np.load("grapping_0.npy") 
    res_zero = pd.DataFrame(res_zero)
    res_zero.columns = ["x", "y", "z", "ro", "pi", "ya", "gripper"]
    #print("0: ", res_zero)
    res_zero_mean = res_zero.mean(axis=0)
    res_zero_mean["gripper"] = round(res_zero_mean["gripper"])
    print(res_zero_mean.to_frame().T)
    print("===================================================")

    res_p10 = np.load("grapping_p10.npy") 
    res_p10 = pd.DataFrame(res_p10)
    res_p10.columns = ["x", "y", "z", "ro", "pi", "ya", "gripper"]
    #print("\np10: \n", res_p10)
    res_p10_mean = res_p10.mean(axis=0)
    res_p10_mean["gripper"] = round(res_p10_mean["gripper"])
    print(res_p10_mean.to_frame().T)
    print("===================================================")

    res_r10 = np.load("grapping_r10.npy") 
    res_r10 = pd.DataFrame(res_r10)
    res_r10.columns = ["x", "y", "z", "ro", "pi", "ya", "gripper"]
    #print("\nr10: \n", res_r10)
    res_r10_mean = res_r10.mean(axis=0)
    res_r10_mean["gripper"] = round(res_r10_mean["gripper"])
    print(res_r10_mean.to_frame().T)
    print("===================================================")