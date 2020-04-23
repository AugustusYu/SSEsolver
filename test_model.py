import time
from game_model import *
import scipy.io as scio

def runtime_test_once(k1 = 2, k2  = 2, t  = 3 ):
    tstart = time.time()
    game = Bayesian_stackelberg_game()
    w = np.ones([t])
    w = w/np.sum(w)
    game.set_para(w,k1,k2)

    mlp = MLP_solver()
    A = mlp.solve(game)

    dob = DOBSS_solver()
    B = dob.solve(game)

    HBGS = HBGS_solver()
    C = HBGS.solve_depthone(game)
    D = HBGS.solve_binary_tree(game)

    time_once = np.array([A[1],B,C[1],D[1]])
    return time_once

def runtime_test_2():
    # type = np.arange(2,11)
    # fa = np.arange(2,11)
    # la = np.arange(2,11)

    # type = np.array([5])
    # fa = np.arange(2,20,2)
    # la = np.arange(2, 20, 2)

    type = np.arange(3,10)
    fa = np.array([10])
    la = np.array([10])

    outp  = 'runtimetest4.mat'
    data = np.zeros([np.size(type), np.size(la), np.size(fa),4])

    tstart = time.time()
    for t in range (np.size(type)):
        for i in range(np.size(la)):
            j = i
            #for j in range (np.size(fa)):
            A = runtime_test_once(la[i],fa[j],type[t])
            data[t,i,j,:] = A
            print('test for ', [t, i, j], ' result is ', A)
            print('current time is ',time.time()-tstart)
            scio.savemat(outp, {'type':type, 'fa':fa,'la':la,'data':data})

def runtime_time_repeat()
    type = np.array([4])
    fa = np.arange(5,11)
    la = fa
    Nrepeat = 10

    outp  = '0420runtimetest2.mat'
    data = np.zeros([np.size(type), np.size(la), np.size(fa),4, Nrepeat])
    err_time = np.zeros([np.size(type), np.size(la), np.size(fa), 2,4,Nrepeat])
    exp_size = np.zeros([np.size(type), np.size(la), np.size(fa), 3])

    tstart = time.time()
    for t in range (np.size(type)):
        for i in range(np.size(la)):
            for iter in range (Nrepeat):
                j = i
                #for j in range (np.size(fa)):
                ans = runtime_test_once_upper_bound(la[i],fa[j],type[t])
                data[t,i,j,:, iter] = ans[0]
                err_time[t,i,j,:,:,iter] = ans[2]
                exp_size[t,i,j,:] = ans[1]
                print('test for ', [t, i, j], ' result is ', ans[0])
                print('current time is ',time.time()-tstart)
                scio.savemat(outp, {'type':type, 'fa':fa,'la':la,'data':data,'exp_num':exp_size, 'err_time':err_time})

def runtime_test_once_upper_bound(k1 = 2, k2  = 2, t  = 3 ):
    tstart = time.time()
    game = Bayesian_stackelberg_game()
    w = np.ones([t])
    w = w/np.sum(w)
    game.set_para(w,k1,k2)
    errlist = [0.01,0.03,0.1,0.3]
    err_time = np.zeros([2,4])
    exp_size = np.zeros(3)

    mlp = MLP_solver()
    A = mlp.solve(game)
    #
    dob = DOBSS_solver()
    B = dob.solve(game)

    HBGS = HBGS_solver_upper_bound()
    C = HBGS.solve_depthone_upper_bound(game,errlist)
    D = HBGS.solve_binary_upper_bound(game,errlist)
    init_time = D[3]

    exp_size = np.array([pow(k1,t), C[1], D[1]])
    err_time[0,:] = C[2]
    err_time[1,:] = D[2] + init_time

    time_once = np.array([A[1],B,C[0],D[0]+init_time])

    print(time_once, exp_size, err_time)
    return time_once, exp_size, err_time

def performance_test(k1,k2,t):
    tstart = time.time()
    game = Bayesian_stackelberg_game()
    w = np.ones([t])
    w = w/np.sum(w)
    game.set_para(w,k1,k2)
    iter_list = np.arange(1,501)

    u = np.zeros([3, np.size(iter_list)])

    HBGS = HBGS_solver_upper_bound()
    C = HBGS.solve_depthone_upper_bound_iter(game,iter_list)
    D = HBGS.solve_binary_upper_bound_iter(game,iter_list)

    eraser = sample_model_eraser()
    E = eraser.solve_sample_iterlist(game, iter_list)

    u[0,:] = C
    u[1, :] = D
    u[2, :] = E
    print(u)
    return u

def sample_test():
    Ntype = 4
    fa = 10
    la = fa
    Nrepeat = 10

    data = np.zeros([3,500, Nrepeat])
    outp = '0420approtest3.mat'
    tstart = time.time()
    for iter in range (Nrepeat):
        ans = performance_test(fa,la,Ntype)
        data[:,:,iter] = ans
        print('test for iter = ', iter)
        print('current time is ', time.time() - tstart)
        scio.savemat(outp, {'data':data})


if __name__ == '__main__':
    print('begin game test')
    sample_test()


