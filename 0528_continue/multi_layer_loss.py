
import torch
import torch.nn as nn

class MultiLayerLoss(nn.Module):
    def __init__(self,total_level=3, alpha=1, beta=0.8, p_loss=3.5,device='cpu'):
        super(MultiLayerLoss, self).__init__()
        self.alpha = alpha # Weight for the incorrect classification loss
        self.beta = beta # Weight for the mother layer classification loss
        self.total_level = total_level # Total number of levels in the hierarchy
        self.hierarchy_dict = {
                'layer1':{
                    0:[0],
                    1:[1,2]
                },
                'layer2':{
                    0:[0],
                    1:[1,3],
                    2:[2]
                }
            }
        self.p_loss = p_loss # Weight for the dependence loss
        self.device = device # Device to run the model on   
    
    def calculate_lloss(self, predictions, true_labels): # 计算每一层分错导致的损失       
        lloss = 0
        for l in range(self.total_level):
            lloss += nn.CrossEntropyLoss()(predictions[l], true_labels[l].long())
        return self.alpha * lloss
    

    def check_hierarchy(self, level, current_level, previous_level):
        # 根据level获取对应的字典
        level_dict = self.hierarchy_dict.get(f'layer{level}', None)
        if level_dict is None:
            raise ValueError(f"Invalid level: {level}")

        # 创建一个新的张量来存储结果
        result = torch.zeros_like(current_level)

        # 对current_level和previous_level中的每一对值进行检查
        for i, (curr, prev) in enumerate(zip(current_level, previous_level)):
            # 获取previous_level的值
            previous_level_values = level_dict.get(prev.item(), None)
            if previous_level_values is None:
                raise ValueError(f"Invalid previous level: {prev.item()}")

            # 检查current_level是否在previous_level_values中
            if curr.item() not in previous_level_values:
                result[i] = 1

        return result
    
    def calculate_dloss(self, predictions, true_labels):
        dloss = 0
        epsilon = 1e-7  # 防止除以零
        f1_weight = 0  # F1分数的权重
        f2_weight = 0  # F2分数的权重
        f3_weight = 0  # F3分数的权重
        # f_weight = 10
        for l in range(1, self.total_level):
            current_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l]), dim=1) #当前层的预测结果
            prev_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l-1]), dim=1) #上一层的预测结果
            D_l = self.check_hierarchy(l,current_lvl_pred, prev_lvl_pred) #检查上一层和这一层是否属于同一类，如果不是返回1，是返回0
            l_prev = torch.where(prev_lvl_pred == true_labels[l-1], torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))
            #如果上一层的预测结果和真实标签相同，返回0，否则返回1

            l_curr = torch.where(current_lvl_pred == true_labels[l], torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))
            #如果当前层的预测结果和真实标签相同，返回0，否则返回1

            # 计算F1和F2分数
            tp = torch.sum((true_labels[l] == 1) & (current_lvl_pred == 1))
            fp = torch.sum((true_labels[l] == 1) & (current_lvl_pred == 0))
            fn = torch.sum((true_labels[l] == 0) & (current_lvl_pred == 1))
            f1 = 2*tp / (2*tp + fp + fn + epsilon)
            # f = (fn+fp)/(100*tp+fn+fp+epsilon)

            tp = torch.sum((true_labels[l] == 1) & (current_lvl_pred == 1))
            fp = torch.sum((true_labels[l] == 2) & (current_lvl_pred == 1))
            fn = torch.sum((true_labels[l] == 1) & (current_lvl_pred == 2))
            f2 = 2*tp / (2*tp + fp + fn + epsilon)

            # 计算F3分数
            tp = torch.sum((true_labels[l] == 3) & (current_lvl_pred == 3))
            fp = torch.sum((true_labels[l] == 3) & (current_lvl_pred == 1))
            fn = torch.sum((true_labels[l] == 1) & (current_lvl_pred == 3))
            f3 = 2*tp / (2*tp + fp + fn + epsilon)

            # 将F1、F2和F3分数纳入损失函数
            dloss += torch.sum(torch.pow(self.p_loss, D_l*l_prev)*torch.pow(self.p_loss, D_l*l_curr) - 1)- f1_weight * f1 - f2_weight * f2 - f3_weight * f3
            # dloss += torch.sum(torch.pow(self.p_loss, D_l*l_prev)*torch.pow(self.p_loss, D_l*l_curr) - 1) +f*f_weight- f1_weight * f1 - f2_weight * f2 - f3_weight * f3

        return self.beta * dloss