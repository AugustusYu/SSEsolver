import numpy as np
import time
from pulp import *
from os import system

class Bayesian_stackelberg_game():
    leader = 1
    follower_type = 2
    type_w = np.zeros([follower_type])
    leader_action = 2
    follower_action = 2
    leader_utility = np.zeros([follower_type, leader_action, follower_action])
    follower_utility = np.zeros([follower_type, leader_action, follower_action])

    def __init__(self):
        #print('a game model is initialized')
        pass

    def __call__(self):
        print('a game model is called')

    def set_para(self, w, a1, a2):
        self.type_w = w.copy()
        self.follower_type = np.size(w)
        self.leader_action = a1
        self.follower_action = a2
        self.random_utility()

    def set_utility(self,u1,u2):
        self.leader_utility = u1.copy()
        self.follower_utiltiy = u2.copy()

    def random_utility(self):
        my_seed = 7
        #np.random.seed(my_seed)
        self.leader_utility = np.random.rand(self.follower_type, self.leader_action, self.follower_action)
        #np.random.seed(my_seed * 3)
        self.follower_utility = np.random.rand(self.follower_type, self.leader_action, self.follower_action)

    def out_utility(self):
        return [self.leader_utility, self.follower_utility]

    def out_type(self):
        return self.type_w

    def binary_split_game(self):
        child1 = Bayesian_stackelberg_game()
        child2 = Bayesian_stackelberg_game()
        if self.follower_type > 1 :
            pos = int(self.follower_type / 2)
            w = self.type_w.copy()

            child1.set_para(w[:pos]/np.sum(w[:pos]),self.leader_action, self.follower_action)
            child1.set_utility(self.leader_utility[:pos, :,:], self.follower_utility[:pos,:,:])
            child2.set_para(w[pos:] / np.sum(w[pos:]), self.leader_action, self.follower_action)
            child2.set_utility(self.leader_utility[pos:, :, :], self.follower_utility[pos:, :, :])
        else:
            print(' this is a single game ')


        return child1, child2

class DOBSS_solver():
    #game = Bayesian_stackelberg_game()

    def __init__(self):
        #print('use DOBSS to solve game')
        pass

    def solve(self,game):
        u = game.out_utility()
        lu = u[0]
        fu = u[1]
        w = game.out_type()
        N_type = lu.shape[0]
        N_la = lu.shape[1]
        N_fa = lu.shape[2]
        M = N_type * N_la * N_fa

        tstart = time.time()
        prob = pulp.LpProblem(sense=LpMaximize)

        z = LpVariable.dicts(name='z', indexs=range(N_type * N_la * N_fa), lowBound=0,upBound=1, cat=LpContinuous)
        q = LpVariable.dicts(name='q', indexs=range(N_type *  N_fa), cat=LpBinary)
        a = LpVariable.dicts(name='a', indexs=range(N_type), lowBound=0, cat=LpContinuous)

        uz = np.zeros([N_type * N_la * N_fa])
        for l in range(N_type):
            for i in range (N_la):
                for j in range (N_fa):
                    uz[l*N_la*N_fa + i*N_fa + j] = lu[l,i,j]*w[l]
        prob += lpSum ( z[i] * uz[i] for i in range (N_type * N_la * N_fa))

        for l in range (N_type):
            prob += lpSum ( z[i+l*N_la*N_fa] for i in range (N_la*N_fa) ) == 1

            for j in range (N_fa):
                prob += q[l*N_fa +j] <= lpSum( z[l*N_la*N_fa + i*N_fa + j] for i in range (N_la))

            prob += lpSum( q[l*N_fa +j] for j in range (N_fa) ) == 1

            for i in range (N_la):
                prob += lpSum( z[l*N_la*N_fa + i*N_fa + j] for j in range (N_fa) ) \
                        == lpSum( z[0*N_la*N_fa + i*N_fa + j] for j in range (N_fa) )

            for j in range(N_fa):
                #print(N_fa, N_la, N_type,l,j)
                prob += lpSum( [z[l*N_la*N_fa + int(np.floor(zi/N_fa))*N_fa + zi%N_fa]
                                * fu[l,int(np.floor(zi/N_fa)),j] for zi in range (N_la*N_fa) ]) <= a[l]
                prob +=  lpSum(a[l] - [z[l*N_la*N_fa + int(np.floor(zi/N_fa))*N_fa + zi%N_fa]
                                * fu[l,int(np.floor(zi/N_fa)),j] for zi in range (N_la*N_fa) ] ) <= (1 - q[l*N_fa +j]) *M

        prob.solve()
        # print('DOBSS solve time is ', time.time()-tstart, 'seconds')
        return time.time()-tstart

class MLP_solver():
    #game = Bayesian_stackelberg_game()

    def __init__(self):
        #print('use mlp to solve game')
        pass

    def solve(self,game):
        u = game.out_utility()
        lu = u[0]
        fu = u[1]
        w = game.out_type()
        Ntype = lu.shape[0]
        N_la = lu.shape[1]
        N_fa = lu.shape[2]

        Nmax_action = pow(N_fa, Ntype)
        # since it's an exp space, we don't run them all
        random_sample = 100
        tstart = time.time()
        for iter in range(random_sample):
            this_action = np.random.randint(Nmax_action)
            #print('this action is ', this_action)
            action_list = self.action_set(this_action, Ntype, N_fa)
            self.sub_lp(lu,w,Ntype,action_list, N_la, fu, N_fa)
        tend = time.time()
        # print('for %d iter, we need '%random_sample, tend-tstart, ' seconds')
        # print('expected iter is ', Nmax_action)
        # print('expected time is ', Nmax_action / random_sample * (tend-tstart), ' seconds')
        return [Nmax_action,  Nmax_action / random_sample * (tend-tstart)]

    def action_set(self,n_action,n_type,na):
        a = np.zeros([n_type], dtype = int)
        b = n_action
        for i in range (n_type):
            a[i] = b % na
            b = np.floor(b/na)
        return a

    def sub_lp(self,lu,w,ntype, fa,nla, fu ,nfa):
        prob = pulp.LpProblem(sense = LpMaximize)
        x = LpVariable.dicts(name='x', indexs=range(nla), lowBound=0, cat=LpContinuous)
        ua = np.zeros([nla])
        for i in range (nla):
            for j in range(ntype):
                ua[i] += lu[j,i,fa[j]] * w[j]
        prob += lpSum(x[i] * ua[i] for i in range (nla))
        for j in range(ntype):
            for k in range (nfa):
                if k != fa[j]:
                    #print('lpsum is ', lpSum(x[i] * (fu[j, i, fa[j]] - fu[j, i, k]) for i in range(nla)))
                    prob += lpSum( [ x[i]*fu[j,i,fa[j]] for i in range (nla) ] ) >= lpSum( [ x[i]*fu[j,i,k] for i in range (nla) ] )
        prob += lpSum(x) == 1
        prob.solve()

class HBGS_solver():
    # game = Bayesian_stackelberg_game()
    # this older solver

    def __init__(self):
        #print('use hbgs to solve game')
        pass

    def solve_depthone(self, game):
        u = game.out_utility()
        lu = u[0]
        fu = u[1]
        w = game.out_type()
        Ntype = lu.shape[0]
        Nla = lu.shape[1]
        Nfa = lu.shape[2]
        M = 10.0*Nla*Nfa*Ntype  # an upper bound for


        feasible_action = self.init_feasible_action_depth_one( Ntype, Nfa, fu)
        feasible_num = np.sum(feasible_action, axis=1)
        Nmax_action = 1
        for i in range (Ntype):
            Nmax_action = Nmax_action * feasible_num[i]
        max_r = 0  # current reward
        upper_b = np.ones([int(Nmax_action)] ,dtype=float) * M  #upper bound for each action
        max_alpha = np.zeros([Nla])
        random_sample = 100

        tstart = time.time()
        for iter in range(random_sample):
            this_action = np.argmax(upper_b)
            action_list = self.action_set_feasible(this_action, Ntype, feasible_num)
            r = self.sub_lp(lu,w,Ntype,action_list, Nla, fu, Nfa)
            # upper_b[this_action] = r
            # if r > max_r:
            #     max_r = r
        tend = time.time()
        # print('for %d iter, we need '%random_sample, tend-tstart, ' seconds')
        # print('expected iter is ', Nmax_action)
        # print('expected time is ', Nmax_action / random_sample * (tend-tstart), ' seconds')
        return [Nmax_action, Nmax_action / random_sample * (tend-tstart)]

    def sub_lp(self,lu,w,ntype, fa,nla, fu ,nfa, status=True):
        prob = pulp.LpProblem(sense = LpMaximize)
        x = LpVariable.dicts(name='x', indexs=range(nla), lowBound=0, cat=LpContinuous)
        ua = np.zeros([nla])
        for i in range (nla):
            for j in range(ntype):
                ua[i] += lu[j,i,fa[j]] * w[j]
        prob += lpSum(x[i] * ua[i] for i in range (nla))
        for j in range(ntype):
            for k in range (nfa):
                if k != fa[j]:
                    #print('lpsum is ', lpSum(x[i] * (fu[j, i, fa[j]] - fu[j, i, k]) for i in range(nla)))
                    prob += lpSum( [ x[i]*fu[j,i,fa[j]] for i in range (nla) ] ) >= lpSum( [ x[i]*fu[j,i,k] for i in range (nla) ] )
        prob += lpSum(x) == 1
        prob.solve()
        if status:
            return value(prob.objective)
        else:
            return LpStatus[prob.status]


    def action_set_feasible(self,n_action,n_type,na):
        a = np.zeros([int(n_type)], dtype = int)
        b = n_action
        for i in range (n_type):
            t = n_type-i-1
            a[t] = b % na[t]
            b = np.floor(b/na[t])
        return a

    def init_upper_bound_depth_one(self):
        y = np.zeros([Ntype, Nfa])
        for i in range (Ntype):
            for j in range (Nfa):
                y[i,j] = np.max(lu[i,:,j])
        return y

    def init_feasible_action_depth_one(self, Ntype, Nfa, fu):
        feasible_num = np.ones([Ntype],dtype=int) * Nfa
        feasible_action = np.zeros([Ntype, Nfa])
        for type in range (Ntype):
            for i in range (Nfa):
                flag = True
                for j in range (Nfa):
                    if j != i:
                        if (fu[type,:,j] > fu[type,:,i] ).all():
                            flag = False
                            break
                if flag == True:
                    feasible_action[type,i] = 1
                else:
                    feasible_action[type, i] = 0
        return feasible_action

    def solve_binary_tree(self, game):
        u = game.out_utility()
        lu = u[0]
        fu = u[1]
        w = game.out_type()
        Ntype = lu.shape[0]
        Nla = lu.shape[1]
        Nfa = lu.shape[2]
        M = 10.0*Nla*Nfa*Ntype  # an upper bound for

        tstart = time.time()
        # feasible_action = self.binary_feasible_and_boundry(game)

        [child1, child2] = game.binary_split_game()
        f1 = self.sub_game_feasible(child1)
        f2 = self.sub_game_feasible(child2)
        tend_init = time.time()
        Nmax_action = pow(Nfa, Ntype)
        random_sample = 100
        for iter in range(random_sample):
            this_action = np.random.randint(Nmax_action)
            #print('this action is ', this_action)
            action_list = self.action_set(this_action, Ntype, Nfa)
            self.sub_lp(lu,w,Ntype,action_list, Nla, fu, Nfa)
        tend = time.time()
        # print('for %d iter, we need '%random_sample, tend-tstart, ' seconds')
        Nmax_action = f1 * f2
        # print('expected iter is ', Nmax_action)
        expt = Nmax_action / random_sample * (tend-tend_init) + (tend_init-tstart)
        # print('expected time is ', expt, ' seconds')
        return [Nmax_action,  expt]

    def sub_game_feasible(self,game):
        def action_set( n_action, n_type, na):
            a = np.zeros([n_type], dtype=int)
            b = n_action
            for i in range(n_type):
                a[i] = b % na
                b = np.floor(b / na)
            return a

        u = game.out_utility()
        lu = u[0]
        fu = u[1]
        w = game.out_type()
        Ntype = lu.shape[0]
        N_la = lu.shape[1]
        N_fa = lu.shape[2]

        Nmax_action = pow(N_fa, Ntype)
        # since it's an exp space, we don't run them all
        feasible_num = 0
        tstart = time.time()
        for iter in range(Nmax_action):
            this_action = iter
            action_list = action_set(this_action, Ntype, N_fa)
            a = self.sub_lp(lu,w,Ntype,action_list, N_la, fu, N_fa,False)
            if a == 'Infeasible':
                pass
            else:
                feasible_num += 1
        tend = time.time()
        # print('sub game feasible need ', tend-tstart, ' seconds')
        # print('this sub game has %d feasible solutions '%feasible_num)
        return feasible_num


    def exp_feasible_binary(self):
        pass
        print('not sure how to calculate such case')


    def binary_feasible_and_boundry(self, game):
        u = game.out_utility()
        lu = u[0]
        fu = u[1]
        w = game.out_type()
        Ntype = lu.shape[0]
        Nla = lu.shape[1]
        Nfa = lu.shape[2]

        feasible_action = np.zeros([Ntype, Nfa])
        if Ntype>=2:
            [child1, child2] = game.binary_split_game()
            faction1 = self.binary_feasible_and_boundry(child1)
            faction2 = self.binary_feasible_and_boundry(child2)
            # this function is not finished
        else:
            for i in range(Nfa):
                flag = True
                for j in range(Nfa):
                    if j != i and (fu[0,:,j] > fu[0,:,i] ).all():
                        flag = False
                        break
                if flag == True:
                    feasible_action[0, i] = 1
                else:
                    feasible_action[0, i] = 0

        return feasible_action

    def action_set(self,n_action,n_type,na):
        a = np.zeros([n_type], dtype = int)
        b = n_action
        for i in range (n_type):
            a[i] = b % na
            b = np.floor(b/na)
        return a

class HBGS_solver_upper_bound():

    def __init__(self):
        #print('use hbgs to solve game')
        pass

    def sub_lp(self,lu,w,ntype, fa,nla, fu ,nfa, status=True):
        prob = pulp.LpProblem(sense = LpMaximize)
        x = LpVariable.dicts(name='x', indexs=range(nla), lowBound=0, cat=LpContinuous)
        ua = np.zeros([nla])
        for i in range (nla):
            for j in range(ntype):
                ua[i] += lu[j,i,fa[j]] * w[j]
        prob += lpSum(x[i] * ua[i] for i in range (nla))
        for j in range(ntype):
            for k in range (nfa):
                if k != fa[j]:
                    #print('lpsum is ', lpSum(x[i] * (fu[j, i, fa[j]] - fu[j, i, k]) for i in range(nla)))
                    prob += lpSum( [ x[i]*fu[j,i,fa[j]] for i in range (nla) ] ) >= lpSum( [ x[i]*fu[j,i,k] for i in range (nla) ] )
        prob += lpSum(x) == 1
        prob.solve()
        if LpStatus[prob.status] == 'Optimal':
            flag = True
        else:
            flag = False
        return flag, value(prob.objective)

    def sub_lp_depth_one(self, ux, fa, nfa, fu):
        prob = pulp.LpProblem(sense = LpMaximize)
        nla = len(ux)
        x = LpVariable.dicts(name='x', indexs=range(nla), lowBound=0, cat=LpContinuous)
        prob += lpSum(x[i] * ux[i] for i in range(nla))
        for k in range(nfa):
            if k != fa:
                prob += lpSum(x[i]*fu[i,fa] for i in range(nla) ) >= lpSum(x[i]*fu[i,k] for i in range(nla) )
        prob += lpSum(x) == 1
        prob.solve()
        flag = True
        if LpStatus[prob.status] == 'Optimal':
            flag = True
        else:
            flag = False

        return flag, value(prob.objective)


    def solve_depthone_upper_bound(self, game, errlist = 0):
        u = game.out_utility()
        lu = u[0]
        fu = u[1]
        w = game.out_type()
        Ntype = lu.shape[0]
        Nla = lu.shape[1]
        Nfa = lu.shape[2]
        M = 10.0 * Nla * Nfa * Ntype  # an upper bound for

        action_info = self.init_feasible_action_depth_one(Ntype, Nfa, lu, fu)
        action_flag = action_info[0]
        action_upper = action_info[1]

        #print('finish initial')

        feasible_num = np.sum(action_flag, axis=1)

        Nmax_action = 1
        for i in range(Ntype):
            Nmax_action = Nmax_action * feasible_num[i]
        max_r = 0  # current reward
        upper_b = np.ones([int(Nmax_action)], dtype=float) * M  # upper bound for each action
        max_alpha = np.zeros([Nla])
        random_sample = 100


        ### exp feasible time
        # tstart = time.time()
        # for iter in range(random_sample):
        #     this_action = np.argmax(upper_b)
        #     action_list = self.action_set_feasible(this_action, Ntype, feasible_num)
        #     r = self.sub_lp(lu, w, Ntype, action_list, Nla, fu, Nfa)
        # tend = time.time()
        # exp_time = Nmax_action / random_sample * (tend - tstart)

        ### upper bound time with feasible, error >= 0
        tstart = time.time()
        now_reward = 0
        action_bound = np.zeros([pow(Nfa,Ntype)])
        for num in range ( pow(Nfa,Ntype) ):
            fa = self.num2action(Nfa,Ntype,num)

            a = 0
            for i in range (Ntype):
                a += action_upper[i,fa[i]] * w[i]
            #print(num, fa,a)
            action_bound[num] = a
        cur_u = 0
        max_u = np.max(action_bound)
        maxiter = 1000
        iter = 0
        err_len = len(errlist)
        err_time = np.zeros(err_len)
        err_flag = np.zeros(err_len)
        while iter < maxiter and cur_u < max_u :
            idx = np.argmax(action_bound)
            action = self.num2action(Nfa,Ntype,idx)
            ans = self.sub_lp(lu, w, Ntype, action, Nla, fu, Nfa)
            if ans[0]>0:
                action_bound[idx] = ans[1]
                if ans[1] > cur_u:
                    cur_u = ans[1]
            else:
                action_bound[idx] = -100
            max_u = np.max(action_bound)

            for i in range ( err_len):
                if err_flag[i] == 0 and cur_u >= max_u * (1-errlist[i]):
                    err_flag[i] = 1
                    err_time[i] = time.time()-tstart

            iter += 1
            # print('iter=',iter, ' cur_u=',cur_u, ' max_u=',max_u, ' time=',time.time()-tstart)
            # print(action_bound, action,idx)

        #print('finish upper bound iteration, with iter=',iter, ' cur_u=',cur_u, ' max_u=',max_u )
        tend = time.time()
        upper_bound_time = tend-tstart
        #print('optimal time=', upper_bound_time, ' error_time=',err_time)
        return upper_bound_time,Nmax_action, err_time

    def solve_depthone_upper_bound_iter(self, game, iter_list = 0):
        u = game.out_utility()
        lu = u[0]
        fu = u[1]
        w = game.out_type()
        Ntype = lu.shape[0]
        Nla = lu.shape[1]
        Nfa = lu.shape[2]
        M = 10.0 * Nla * Nfa * Ntype  # an upper bound for

        action_info = self.init_feasible_action_depth_one(Ntype, Nfa, lu, fu)
        action_flag = action_info[0]
        action_upper = action_info[1]

        #print('finish initial')

        feasible_num = np.sum(action_flag, axis=1)

        Nmax_action = 1
        for i in range(Ntype):
            Nmax_action = Nmax_action * feasible_num[i]
        max_r = 0  # current reward
        upper_b = np.ones([int(Nmax_action)], dtype=float) * M  # upper bound for each action
        max_alpha = np.zeros([Nla])
        random_sample = 100


        ### exp feasible time
        # tstart = time.time()
        # for iter in range(random_sample):
        #     this_action = np.argmax(upper_b)
        #     action_list = self.action_set_feasible(this_action, Ntype, feasible_num)
        #     r = self.sub_lp(lu, w, Ntype, action_list, Nla, fu, Nfa)
        # tend = time.time()
        # exp_time = Nmax_action / random_sample * (tend - tstart)

        ### upper bound time with feasible, error >= 0
        tstart = time.time()
        now_reward = 0
        action_bound = np.zeros([pow(Nfa,Ntype)])
        for num in range ( pow(Nfa,Ntype) ):
            fa = self.num2action(Nfa,Ntype,num)

            a = 0
            for i in range (Ntype):
                a += action_upper[i,fa[i]] * w[i]
            #print(num, fa,a)
            action_bound[num] = a
        cur_u = 0
        max_u = np.max(action_bound)
        maxiter = 1100
        iter = 0
        err_len = len(iter_list)
        err_time = np.zeros(err_len)
        err_flag = np.zeros(err_len)
        while iter < maxiter :
            idx = np.argmax(action_bound)
            action = self.num2action(Nfa,Ntype,idx)
            ans = self.sub_lp(lu, w, Ntype, action, Nla, fu, Nfa)
            if ans[0]>0:
                action_bound[idx] = ans[1]
                if ans[1] > cur_u:
                    cur_u = ans[1]
            else:
                action_bound[idx] = -100
            max_u = np.max(action_bound)

            for i in range ( err_len):
                if err_flag[i] == 0 and iter == iter_list[i]:
                    err_flag[i] = 1
                    err_time[i] = cur_u

            iter += 1
            # print('iter=',iter, ' cur_u=',cur_u, ' max_u=',max_u, ' time=',time.time()-tstart)
            # print(action_bound, action,idx)

        #print('finish upper bound iteration, with iter=',iter, ' cur_u=',cur_u, ' max_u=',max_u )
        tend = time.time()
        upper_bound_time = tend-tstart
        #print('optimal time=', upper_bound_time, ' error_time=',err_time)
        return err_time


    def action2num(self, Nfa, Ntype, action):
        s = 0
        for k in range (Ntype):
            s = s*Nfa + action[k]
        return s


    def num2action(self, Nfa, Ntype, num):
        action = np.zeros(Ntype, dtype=int)
        a = num
        for k in range (Ntype):
            action[Ntype-k-1] = np.mod(a, Nfa)
            a = np.floor(a / Nfa)
        return action

    def init_feasible_action_depth_one(self,Ntype, Nfa, lu, fu):
        action_flag = np.zeros([Ntype, Nfa ]) # whether action is feasible
        action_upper = np.zeros([Ntype, Nfa]) # upper bound
        Nla = lu.shape[1]
        for ft in range(Ntype):
            for fa in range(Nfa):
                ans = self.sub_lp_depth_one(lu[ft,:,fa].reshape([Nla])  , fa, Nfa, fu[ft,:,:].reshape([Nla,Nfa]))
                action_flag[ft,fa] = ans[0]
                if ans[0] > 0:
                    action_upper[ft, fa] = ans[1]
                else:
                    action_upper[ft,fa ] = -100
        #print(action_flag)
        #print(action_upper)
        return action_flag, action_upper

    def solve_binary_upper_bound(self, game, errlist = 0):
        u = game.out_utility()
        lu = u[0]
        fu = u[1]
        w = game.out_type()
        Ntype = lu.shape[0]
        Nla = lu.shape[1]
        Nfa = lu.shape[2]
        M = 10.0 * Nla * Nfa * Ntype  # an upper bound for

        left_type = int(np.floor(Ntype/2))
        right_type = Ntype - left_type
        #print(left_type, right_type)
        left_w = w[:left_type] / np.sum(w[:left_type] )
        right_w =  w[left_type:] / np.sum(w[left_type:] )

        tstart = time.time()
        left_action_info = self.init_feasible_action_whole_tree(left_type, Nfa, lu, fu,left_w)
        right_action_info = self.init_feasible_action_whole_tree(right_type, Nfa, lu, fu, right_w)
        init_time = time.time()-tstart
        #print('binary init time is ', init_time)
        Nmax_action = np.sum(left_action_info[0] > 0.1) * np.sum(right_action_info[0] > 0.1)
        #print('Nmax action is ', Nmax_action)

        tstart = time.time()
        action_bound = np.zeros([pow(Nfa,Ntype)])
        for num in range ( pow(Nfa,Ntype) ):
            fa = self.num2action(Nfa,Ntype,num)
            left_num = self.action2num(Nfa, left_type, fa[:left_type])
            right_num = self.action2num(Nfa, right_type, fa[right_type:])

            a = left_action_info[1][left_num] * left_type / Ntype \
                +right_action_info[1][right_num] * right_type / Ntype
            #print(num, fa,a)
            action_bound[num] = a

        cur_u = 0
        max_u = np.max(action_bound)
        maxiter = 1000
        iter = 0
        err_len = len(errlist)
        err_time = np.zeros(err_len)
        err_flag = np.zeros(err_len)
        while iter < maxiter and cur_u < max_u :
            idx = np.argmax(action_bound)
            action = self.num2action(Nfa,Ntype,idx)
            ans = self.sub_lp(lu, w, Ntype, action, Nla, fu, Nfa)
            if ans[0]>0:
                action_bound[idx] = ans[1]
                if ans[1] > cur_u:
                    cur_u = ans[1]
            else:
                action_bound[idx] = -100
            max_u = np.max(action_bound)

            for i in range ( err_len):
                if err_flag[i] == 0 and cur_u >= max_u * (1-errlist[i]):
                    err_flag[i] = 1
                    err_time[i] = time.time()-tstart

            iter += 1
            # print('iter=',iter, ' cur_u=',cur_u, ' max_u=',max_u, ' time=',time.time()-tstart)
            # print(action_bound, action,idx)

        #print('finish upper bound iteration, with iter=',iter, ' cur_u=',cur_u, ' max_u=',max_u )
        tend = time.time()
        upper_bound_time = tend-tstart
        #print('optimal time=', upper_bound_time, ' error_time=',err_time)

        return upper_bound_time, Nmax_action, err_time, init_time

    def solve_binary_upper_bound_iter(self, game, iter_list = 0):
        u = game.out_utility()
        lu = u[0]
        fu = u[1]
        w = game.out_type()
        Ntype = lu.shape[0]
        Nla = lu.shape[1]
        Nfa = lu.shape[2]
        M = 10.0 * Nla * Nfa * Ntype  # an upper bound for

        left_type = int(np.floor(Ntype/2))
        right_type = Ntype - left_type
        #print(left_type, right_type)
        left_w = w[:left_type] / np.sum(w[:left_type] )
        right_w =  w[left_type:] / np.sum(w[left_type:] )

        tstart = time.time()
        left_action_info = self.init_feasible_action_whole_tree(left_type, Nfa, lu, fu,left_w)
        right_action_info = self.init_feasible_action_whole_tree(right_type, Nfa, lu, fu, right_w)
        init_time = time.time()-tstart
        #print('binary init time is ', init_time)
        Nmax_action = np.sum(left_action_info[0] > 0.1) * np.sum(right_action_info[0] > 0.1)
        #print('Nmax action is ', Nmax_action)

        tstart = time.time()
        action_bound = np.zeros([pow(Nfa,Ntype)])
        for num in range ( pow(Nfa,Ntype) ):
            fa = self.num2action(Nfa,Ntype,num)
            left_num = self.action2num(Nfa, left_type, fa[:left_type])
            right_num = self.action2num(Nfa, right_type, fa[right_type:])

            a = left_action_info[1][left_num] * left_type / Ntype \
                +right_action_info[1][right_num] * right_type / Ntype
            #print(num, fa,a)
            action_bound[num] = a

        cur_u = 0
        max_u = np.max(action_bound)
        maxiter = 1100
        iter = 0
        err_len = len(iter_list)
        err_time = np.zeros(err_len)
        err_flag = np.zeros(err_len)
        while iter < maxiter  :
            idx = np.argmax(action_bound)
            action = self.num2action(Nfa,Ntype,idx)
            ans = self.sub_lp(lu, w, Ntype, action, Nla, fu, Nfa)
            if ans[0]>0:
                action_bound[idx] = ans[1]
                if ans[1] > cur_u:
                    cur_u = ans[1]
            else:
                action_bound[idx] = -100
            max_u = np.max(action_bound)

            for i in range ( err_len):
                if err_flag[i] == 0 and iter == iter_list[i]:
                    err_flag[i] = 1
                    err_time[i] = cur_u

            iter += 1
            # print('iter=',iter, ' cur_u=',cur_u, ' max_u=',max_u, ' time=',time.time()-tstart)
            # print(action_bound, action,idx)

        #print('finish upper bound iteration, with iter=',iter, ' cur_u=',cur_u, ' max_u=',max_u )
        tend = time.time()
        upper_bound_time = tend-tstart
        #print('optimal time=', upper_bound_time, ' error_time=',err_time)

        return err_time


    def init_feasible_action_whole_tree(self, Ntype, Nfa, lu,fu,w ):
        action_upper = np.zeros(pow(Nfa, Ntype)) # upper bound
        action_flag = np.zeros(pow(Nfa, Ntype))
        Nla = Nfa

        for action_num in range(pow(Nfa, Ntype)  ):
            action_list = self.num2action(Nfa, Ntype, action_num)
            ans = self.sub_lp(lu, w, Ntype, action_list, Nla, fu, Nfa)
            action_flag[action_num] = ans[0]
            if ans[0] > 0:
                action_upper[action_num] = ans[1]
            else:
                action_upper[action_num] = -100
        return action_flag, action_upper

class sample_model_eraser():

    def __init__(self):
        #print('use hbgs to solve game')
        pass

    def sample_strategy(self):
        pass


    def solve_sample_errlist(self, game, errlist = 0):

        u = game.out_utility()
        lu = u[0]
        fu = u[1]
        w = game.out_type()
        Ntype = lu.shape[0]
        Nla = lu.shape[1]
        Nfa = lu.shape[2]
        M = 10.0 * Nla * Nfa * Ntype  # an upper bound for

        cur_u = 0
        max_u = +1000
        maxiter = 1000
        iter = 0
        Nmax_action = pow(Nfa, Ntype)

        while iter < maxiter and cur_u < max_u:
            sample_idx = np.random.randint(Nmax_action)
            sample_action = self.num2action(Nfa,Ntype,sample_idx)
            ans = self.sub_lp(lu, w, Ntype, action, Nla, fu, Nfa)
            if ans[0]>0:
                if ans[1] > cur_u:
                    cur_u = ans[1]
            iter += 1

        return cur_u

    def solve_sample_iterlist(self, game, iter_list = 0):

        u = game.out_utility()
        lu = u[0]
        fu = u[1]
        w = game.out_type()
        Ntype = lu.shape[0]
        Nla = lu.shape[1]
        Nfa = lu.shape[2]
        M = 10.0 * Nla * Nfa * Ntype  # an upper bound for

        cur_u = 0
        max_u = +1000
        maxiter = 1000
        iter = 0
        Nmax_action = pow(Nfa, Ntype)
        err_len = len(iter_list)
        err_time = np.zeros(err_len)
        err_flag = np.zeros(err_len)

        while iter < maxiter and cur_u < max_u:
            sample_idx = np.random.randint(Nmax_action)
            sample_action = self.num2action(Nfa,Ntype,sample_idx)
            ans = self.sub_lp(lu, w, Ntype, sample_action, Nla, fu, Nfa)
            if ans[0]>0:
                if ans[1] > cur_u:
                    cur_u = ans[1]
            iter += 1

            for i in range (err_len):
                if err_flag[i] == 0 and iter == iter_list[i]:
                    err_flag[i] = 1
                    err_time[i] = cur_u


        return err_time






    def sub_lp(self,lu,w,ntype, fa,nla, fu ,nfa, status=True):
        prob = pulp.LpProblem(sense = LpMaximize)
        x = LpVariable.dicts(name='x', indexs=range(nla), lowBound=0, cat=LpContinuous)
        ua = np.zeros([nla])
        for i in range (nla):
            for j in range(ntype):
                ua[i] += lu[j,i,fa[j]] * w[j]
        prob += lpSum(x[i] * ua[i] for i in range (nla))
        for j in range(ntype):
            for k in range (nfa):
                if k != fa[j]:
                    #print('lpsum is ', lpSum(x[i] * (fu[j, i, fa[j]] - fu[j, i, k]) for i in range(nla)))
                    prob += lpSum( [ x[i]*fu[j,i,fa[j]] for i in range (nla) ] ) >= lpSum( [ x[i]*fu[j,i,k] for i in range (nla) ] )
        prob += lpSum(x) == 1
        prob.solve()
        if LpStatus[prob.status] == 'Optimal':
            flag = True
        else:
            flag = False
        return flag, value(prob.objective)

    def action2num(self, Nfa, Ntype, action):
        s = 0
        for k in range (Ntype):
            s = s*Nfa + action[k]
        return s


    def num2action(self, Nfa, Ntype, num):
        action = np.zeros(Ntype, dtype=int)
        a = num
        for k in range (Ntype):
            action[Ntype-k-1] = np.mod(a, Nfa)
            a = np.floor(a / Nfa)
        return action




