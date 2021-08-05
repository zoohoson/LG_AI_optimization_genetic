import os
import pandas as pd
import numpy as np
from pathlib import Path

class Simulator:
    def __init__(self):
        
        self.sample_submission = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'sample_submission.csv'))
        self.max_count = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'max_count.csv'))
        self.stock = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'stock.csv'))
        self.order = pd.read_csv(os.path.join(Path(__file__).resolve().parent, 'order.csv'))   
        
        cut = {f'BLK_{i}': 506 if i <= 2 else 400 for i in range(1,5) }
        
        ratio = {}

        ratio['BLK_1'] = {}
        ratio['BLK_1'][4] = 0.851
        ratio['BLK_1'][5] = 0.851
        ratio['BLK_1'][6] = 0.851

        ratio['BLK_2'] = {}
        ratio['BLK_2'][4] = 0.901
        ratio['BLK_2'][5] = 0.901
        ratio['BLK_2'][6] = 0.901

        ratio['BLK_3'] = {}
        ratio['BLK_3'][4] = 0.710
        ratio['BLK_3'][5] = 0.742
        ratio['BLK_3'][6] = 0.759

        ratio['BLK_4'] = {}
        ratio['BLK_4'][4] = 0.700
        ratio['BLK_4'][5] = 0.732
        ratio['BLK_4'][6] = 0.749
        
        self.cut = cut
        self.ratio = ratio
        
        order_dic = { }
        order = self.order

        for time, BLK_1, BLK_2, BLK_3, BLK_4 in zip(order['time'],order['BLK_1'],order['BLK_2'],order['BLK_3'],order['BLK_4']):

            order_count_time = str(pd.to_datetime(time) + pd.Timedelta(hours=18))
            order_dic[order_count_time] = {}

            order_dic[order_count_time][1] = BLK_1
            order_dic[order_count_time][2] = BLK_2
            order_dic[order_count_time][3] = BLK_3
            order_dic[order_count_time][4] = BLK_4
            
        self.order_dic = order_dic
        
    def make_init(self):
        
        PRT_dic = {time : {i : 0 for i in range(1,5)} for time in self.sample_submission['time']}
        MOL_dic = {time : {i : 0 for i in range(1,5)} for time in self.sample_submission['time']}
        BLK_dic = {time : {i : 0 for i in range(1,5)} for time in self.sample_submission['time']}

        ## 4/1일 00:00:00에 기초재고를 추가 
        for i in range(1,5):
            PRT_dic['2020-04-01 00:00:00'][i] = int(self.stock[f'PRT_{i}'])
            MOL_dic['2020-04-01 00:00:00'][i] = int(self.stock[f'MOL_{i}'])
            BLK_dic['2020-04-01 00:00:00'][i] = int(self.stock[f'BLK_{i}'])
            
        self.PRT_dic = PRT_dic
        self.MOL_dic = MOL_dic
        self.BLK_dic = BLK_dic

        
    def cal_prt_mol(self,machine_name):
        
        df = self.df

        ## PRT 개수와 MOL 개수 구하기 
        process_list = []
        for time, event, mol in zip(self.sample_submission['time'],df[f'Event_{machine_name}'],df[f'MOL_{machine_name}']):

            ## check한 몰의 개수만큼 PRT로 
            try:
                val = int(event[-1])
            except:
                pass

            if event == 'PROCESS':
                process_list.append((time,mol,val))

            self.PRT_dic[time][val] += -mol

        for p_start, p_end in zip(process_list[:-48],process_list[48:]):

            p_start_time, p_start_mol, p_start_number = p_start
            p_end_time, p_end_mol, p_end_number = p_end

            self.MOL_dic[p_end_time][p_start_number] += p_start_mol * 0.975
            
            
    def cal_blk(self):
        
        PRT_dic = self.PRT_dic    
        MOL_dic = self.MOL_dic
        BLK_dic = self.BLK_dic
        order_dic = self.order_dic
        ratio = self.ratio
        cut = self.cut
        
        PRT_stock_dic = {time : {i : 0 for i in range(1,5)} for time in self.sample_submission['time']}
        MOL_stock_dic = {time : {i : 0 for i in range(1,5)} for time in self.sample_submission['time']}
        BLK_stock_dic = {time : {i : 0 for i in range(1,5)} for time in self.sample_submission['time']}
        
        blk_diffs = []
        previous_time = [self.sample_submission['time'][0]] + list(self.sample_submission['time'])

        for time, previous in zip(self.sample_submission['time'], previous_time[:-1]):

            month = int(time[6])

            for i in range(1,5):

                if str(time) == '2020-04-01 00:00:00':
                    PRT_stock_dic[time][i] = PRT_dic[time][i]
                    MOL_stock_dic[time][i] = MOL_dic[time][i]
                    BLK_stock_dic[time][i] = BLK_dic[time][i]
                    
                else:
                    PRT_stock_dic[time][i] = PRT_stock_dic[previous][i] + PRT_dic[time][i]
                    MOL_stock_dic[time][i] = MOL_stock_dic[previous][i] + MOL_dic[time][i]
                    BLK_stock_dic[time][i] = BLK_stock_dic[previous][i] + BLK_dic[time][i]

                    if int(time[11:13]) == 18:

                        val = order_dic[time][i]

                        if val > 0 :
                            mol_number = i
                            mol = MOL_stock_dic[time][i]
                            MOL_stock_dic[time][i] = 0

                            blk_gen = int(mol*ratio[f'BLK_{i}'][month]*cut[f'BLK_{i}'])
                            blk_stock = BLK_stock_dic[time][i] + blk_gen
                            blk_diff = blk_stock - val

                            BLK_stock_dic[time][i] = blk_diff
                            blk_diffs.append(blk_diff)
                            
        self.PRT_stock_dic = PRT_stock_dic
        self.MOL_stock_dic = MOL_stock_dic
        self.BLK_stock_dic = BLK_stock_dic
        self.blk_diffs = blk_diffs
        
    def F(self, x, a): return 1 - x/a if x < a else 0
    
    def cal_change_stop_time(self):
        
        df = self.df
        
        change_type = {'A':'', 'B':''}
        change_num = 0
        change_time = 0
        stop_num = 0
        stop_time = 0
        previous_event = {'A':'', 'B':''}
        for row in df.iterrows():
            for machine in ['A', 'B']:
                if 'CHANGE' in row[1][f'Event_{machine}']:
                    change_time += 1
                    if change_type[machine] != row[1][f'Event_{machine}'][-2:]:
                        change_num += 1
                        change_type[machine] = row[1][f'Event_{machine}'][-2:]

                if 'STOP' == row[1][f'Event_{machine}']:
                    stop_time += 1
                    if previous_event[machine] != 'STOP':
                        stop_num += 1

                previous_event[machine] = row[1][f'Event_{machine}']
        return change_time, change_num, stop_time, stop_num
        
    def cal_score(self):
        
        p = 0
        q = 0
        for item in self.blk_diffs:
            if item < 0:
                p = p + abs(item)
            if item > 0:
                q = q + abs(item)

        N = sum([sum(self.order[f'BLK_{i}']) for i in range(1,5)])
        M = len(self.df) * 2
        
        c, c_n, s, s_n = self.cal_change_stop_time()
        
        self.score = 50*self.F(p, 10*N)+20*self.F(q, 10*N)+\
                20*self.F(c, M)/(1+0.1*c_n) + 10*self.F(s, M)/(1 + 0.1*s_n)
        
        self.p = p
        self.q = q
        self.N = N
        self.M = M
        self.c = c
        self.c_n = c_n
        self.s = s
        self.s_n = s_n
        
    def make_stock_df(self):
        
        PRT_l = {i : [] for i in range(1,5)}
        MOL_l = {i : [] for i in range(1,5)}
        BLK_l = {i : [] for i in range(1,5)}

        for time in self.sample_submission['time']:
            for i in range(1,5):
                PRT_l[i].append(self.PRT_stock_dic[time][i])
                MOL_l[i].append(self.MOL_stock_dic[time][i])
                BLK_l[i].append(self.BLK_stock_dic[time][i])
                
                
        df_stock = pd.DataFrame(index = self.sample_submission['time'])

        for i in range(1,5):
            df_stock[f'PRT_{i}'] = PRT_l[i]
        for i in range(1,5):
            df_stock[f'MOL_{i}'] = MOL_l[i]
        for i in range(1,5):
            df_stock[f'BLK_{i}'] = BLK_l[i]
            
        self.df_stock = df_stock             
                
    def get_score(self,df):
        
        self.df = df.copy()
        self.make_init()
        self.cal_prt_mol('A')
        self.cal_prt_mol('B')
        self.cal_blk()
        self.cal_score()
        self.make_stock_df()
        
        return self.score, self.df_stock
