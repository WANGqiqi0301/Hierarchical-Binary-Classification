
import torch
import torch.nn as nn

class MultiLayerLoss(nn.Module):
    def __init__(self,total_level=2, alpha=1, beta=0.8, p_loss=3,device='cpu'):
        super(MultiLayerLoss, self).__init__()
        self.alpha = alpha # Weight for the incorrect classification loss
        self.beta = beta # Weight for the mother layer classification loss
        self.level = total_level # Total number of levels in the hierarchy
        self.hierarchy_dict = {
                'layer1':{
                    0:[0],
                    1:[1,2]
                },
                'layer2':{
                    0:[0],
                    1:[3,4],
                    2:[2]
                }
            }
        self.p_loss = p_loss # Weight for the dependence loss
        self.device = device # Device to run the model on
    def forward(self, predictions, targets):
        # Check for incorrect classification
        incorrect_classification_loss = nn.CrossEntropyLoss()(predictions, targets)

        # Check if leaf layer classification belongs to the mother layers
        mother_layer_targets = self.get_mother_layer_targets(targets)  # Implement get_mother_layer_targets() method
        mother_layer_predictions = self.get_mother_layer_predictions(predictions)  # Implement get_mother_layer_predictions() method
        mother_layer_classification_loss = nn.CrossEntropyLoss()(mother_layer_predictions, mother_layer_targets)

        # Calculate total loss
        total_loss = incorrect_classification_loss + mother_layer_classification_loss

        return total_loss

    
    
    def calculate_lloss(self, predictions, true_labels): # Calculate the layer loss
        lloss = 0
        for l in range(self.total_level):
            lloss += nn.CrossEntropyLoss()(predictions[l], true_labels[l])
        return self.alpha * lloss
    

    def check_hierarchy(self, current_level, previous_level):
        # Check if the predicted class at level l is a children of the class predicted at level l-1 for the entire batch.
        # If the current level's prediction belongs to the superclass (prediction from the prev layer), return 0, else return 1
        bool_tensor = [not current_level[i] in self.hierarchy_dict[previous_level[i].item()] for i in range(previous_level.size()[0])]
        return torch.FloatTensor(bool_tensor).to(self.device)
    
    def calculate_dloss(self, predictions, true_labels):
        dloss = 0
        for l in range(1, self.total_level):
            current_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l]), dim=1) #当前层的预测结果
            prev_lvl_pred = torch.argmax(nn.Softmax(dim=1)(predictions[l-1]), dim=1) #上一层的预测结果
            D_l = self.check_hierarchy(current_lvl_pred, prev_lvl_pred) #检查上一层和这一层是否属于同一类，如果不是返回1，是返回0
            l_prev = torch.where(prev_lvl_pred == true_labels[l-1], torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))
            #如果上一层的预测结果和真实标签相同，返回0，否则返回1

            l_curr = torch.where(current_lvl_pred == true_labels[l], torch.FloatTensor([0]).to(self.device), torch.FloatTensor([1]).to(self.device))
            #如果当前层的预测结果和真实标签相同，返回0，否则返回1

            dloss += torch.sum(torch.pow(self.p_loss, D_l*l_prev)*torch.pow(self.p_loss, D_l*l_curr) - 1)
            #
        return self.beta * dloss
