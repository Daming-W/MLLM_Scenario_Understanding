import os 
from tqdm import tqdm

class Mapper():
    def __init__(self,answer_dict):
        self.answer_dict = answer_dict

        self.bool_dict = {
            "smoking": False,
            "hailing": False,
            "ped_on_lawn": False,
            "crowded": False,
            "fire": False,
            "trash": False,
            "illegal_parking": False,
        }

        # seven predefined scenarios
        self.set_smoking = set([
            "True", "true", "Yes", "yes",
            ])
        self.set_hailing = set([
            "True", "true", "Yes", "yes",
            ])
        self.set_ped_on_lawn = set([
            "True", "true", "Yes", "yes",
            ])
        self.set_crowded = set([
            "True", "true", "Yes", "yes",
            ])
        self.set_fire = set([
            "True", "true", "Yes", "yes",
            ])
        self.set_trash = set([
            "True", "true", "Yes", "yes",
            ])
        self.set_illegal_parking = set([
            "True", "true", "Yes", "yes",
            ])
    
    def sentence2words(self,answer):
        return [word.strip(".,!?") 
                for word in answer.split() 
                if word.strip(".,!?;").isalpha()]

    def answer2bool(self):

        self.bool_dict["smoking"] = any(word in self.set_smoking for word in self.sentence2words(self.answer_dict["smoking"]))

        self.bool_dict["hailing"] = any(word in self.set_hailing for word in self.sentence2words(self.answer_dict["hailing"]))

        self.bool_dict["ped_on_lawn"] = any(word in self.set_ped_on_lawn for word in self.sentence2words(self.answer_dict["ped_on_lawn"]))

        self.bool_dict["crowded"] = any(word in self.set_crowded for word in self.sentence2words(self.answer_dict["crowded"]))

        self.bool_dict["fire"] = any(word in self.set_fire for word in self.sentence2words(self.answer_dict["fire"]))
        
        self.bool_dict["trash"] = any(word in self.set_trash for word in self.sentence2words(self.answer_dict["trash"]))

        self.bool_dict["illegal_parking"] =  any(word in self.set_illegal_parking for word in self.sentence2words(self.answer_dict["illegal_parking"]))

        return self.bool_dict

        

if __name__=="__main__":
    mapper = mapper(answer="True, it is on fire")
