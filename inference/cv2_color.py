import copy
class Color():
    """
    This class defines the colors that we will use during visualize the results. 
    The color defined in RGB 
    Based on the color in: http://tool.oschina.net/commons?type=3
    parameters:
        flag: indicate the mode used in color. rgb or bgr
    """
    def __init__(self, flag='rgb'):
        self.Green = [0, 255, 0]
        self.Blue = [0,0,255]
        self.Cyan = [0,255,255]
        self.MediumSlateBlue = [123,104,238]
        self.Aquamarine = [127,255,212]
        self.PaleGreen = [152,251,152]
        self.ForestGreen = [34,139,34]
        self.Yellow = [255,255,0]
        self.Gold = [255,215,0]
        self.IndianRed = [255,106,106]
        self.Sienna = [255,130,71]
        self.Firebrick = [178,34,34]
        self.Orange = [255,165,0]
        self.OrangeRed = [255,69,0]
        self.Magenta = [255,0,255]
        self.Purple = [160,32,240]

        self.attributes = self.__dict__
        if flag == 'bgr':
            self.color_names, self.color_list = self._get_all_color('bgr')
        else:
            self.color_names, self.color_list = self._get_all_color('rgb')

    
    def _get_all_color(self, flag='rgb'):
        keys = iter(copy.copy(self.attributes))
        color_list = []
        for key in keys:
            if key == 'attributes': # same as the name which is given to the self.__dict__
                continue
            if flag == 'bgr':
                color_list.append(self.attributes[key][::-1])
            else:
                 color_list.append(self.attributes[key])
        
        return keys, color_list
    
    def get_random_color_list(self, number=0):
        """
        number: how many colors will be used in the color
        """
        return self.color_list


 

if __name__ == '__main__':
    color = Color(flag='bgr')
    




