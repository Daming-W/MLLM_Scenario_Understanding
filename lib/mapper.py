import os 
from tqdm import tqdm

class Mapper():

    def __init__(self,answer_dict):
        # input answer sentence dict
        self.answer_dict = answer_dict

        # output boolean dict
        self.bool_dict = {key: None for key in self.answer_dict}

        # keyword for boolean convertions
        self.keyword_dict = set(['True','true','Yes','yes'])

        # define seven scenarios
        self.scenarios = {
            "smoking" : ["smoking"],
            "hailing" : ["hailing"],
            "ped_on_lawn" : ["ped_on_lawn"],
            "crowd" : ["crowd", "tent", "destruction"],
            "fire_or_flood" : ["fire", "flood"],
            "trash_or_fallen_leaves" : ["trash", "fallen_leaves"],
            "illegal_parking" : ["illegal_parking"]
        }
        # output dict
        self.output_dict = {key: False for key in self.scenarios}
    
    def sentence2words(self,answer):
        return [word.strip(".,!?") 
                for word in answer.split() 
                if word.strip(".,!?;").isalpha()]

    def answer2bool(self):

        for sce, answer in self.answer_dict.items():

            self.bool_dict[sce] = any(word in self.keyword_dict for word in self.sentence2words(answer))

        return self.bool_dict

    def merge_bool(self):
        
        # check bool dict first
        if None in self.bool_dict.values():
            print('WARN: None in bool_dict! To use answer2bool to compute bool_dict first!')
            return None
        # convert by predefined scenarios
        for sce, sub_sce_list in self.scenarios.items():
            for sub_sce in sub_sce_list:
                if self.bool_dict[sub_sce] is True:
                    self.output_dict[sce] = True
    
        return self.output_dict

        
        
if __name__=="__main__":
    mapper = mapper(answer="True, it is on fire")
