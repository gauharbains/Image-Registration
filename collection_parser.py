from filepattern import FilePattern, parse_directory,get_matching,parse_filename,get_regex
import os
import itertools


directory_path='/home/gauhar/five_channel_dataset'
pattern="x{xxx}_y{yyy}_z{zzz}_c{ccc}_t{ttt}.ome.tif"

# directory_path='/home/gauhar/four_channel_dataset'
# pattern="x{xxx}_y{yyy}_z{zzz}_c{ccc}.ome.tif"

# directory_path='/home/gauhar/two_channel_dataset'
# pattern="x{xxx}_r{rrr}.ome.tif"



# version 1

# def collection_parser(directory_path,file_pattern,registration_variable):
    
#     # Predefined variables order
#     var_order = 'rtczyx'    
#     file_ind, uvals=parse_directory(directory_path,file_pattern)
#     parser_object=FilePattern(directory_path,file_pattern)
#     reg_ind=[uvals[var] for var in registration_variable ]
#     registration_vars=[char for char in registration_variable]
#     indices_combinations=list(itertools.product(*reg_ind))
#     all_dicts=[]
#     for index_comb in indices_combinations:
#         inter_dict={}
#         for i in range(len(registration_vars)):
#             inter_dict.update({registration_vars[i].upper():index_comb[i]})
#         all_dicts.append(inter_dict)     
 
#     image_sets=[]        
#     for reg_dict in all_dicts:
#         intermediate_set=[]
#         files=parser_object.get_matching(**reg_dict)
#         for file_dict in files:
#             intermediate_set.append(file_dict['file'])
#         image_sets.append(intermediate_set)    
#     return image_sets

# image_sets=collection_parser(directory_path,pattern,'ct')


# version 2    
def collection_parser(directory_path,file_pattern,registration_variable, similarity_variable, template_image):
    
    # Predefined variables order     
    #var_order = 'rtczyx'  
    
    # get all variables in the file pattern
    _,variables=get_regex(file_pattern)
    
    # get variables except the registration and similarity variable
    moving_variables=[var for var in variables if var not in registration_variable and var not in similarity_variable]
        
    # index of all variables in template image
    template_index=parse_filename(template_image,pattern=file_pattern)
    
    #index of the registration variables in the template image
    registration_variable_index=[template_index[var] for var in registration_variable]
    
    #index of the similarity variables in the template image
    similarity_variable_index=[template_index[var] for var in similarity_variable]  
    
    # uvals consists of all the possible index values of all variables
    file_ind, uvals=parse_directory(directory_path,file_pattern)    

    parser_object=FilePattern(directory_path,file_pattern)     
    
    image_set=[]
    
    # get all the permutations of the moving variables
    moving_variables_set=[uvals[var] for var in moving_variables]
    
    # append the fixed value of the similarity variable
    for char in similarity_variable:
        moving_variables.append(char)       
        
        for ind in uvals[char]:
            registration_set=[]
            
            moving_variables_set.append([ind])   
            
            registration_indices_combinations=list(itertools.product(*moving_variables_set))     
            
            all_dicts=[]
            for index_comb in registration_indices_combinations:
                inter_dict={}
                for i in range(len(moving_variables)):
                    inter_dict.update({moving_variables[i].upper():index_comb[i]})
                all_dicts.append(inter_dict)           
                    
            for reg_dict in all_dicts:
                intermediate_set=[]
                files=parser_object.get_matching(**reg_dict)
                for file_dict in files:
                    intermediate_set.append(file_dict['file'])
                registration_set.append(intermediate_set) 
            
            moving_variables_set.pop(-1)
            image_set.append(registration_set)
        
    return image_set

template_image='x001_y001_z001_c001_t001.ome.tif'
# template_image='x001_y001_z001_c001.ome.tif'
# template_image='x001_r001.ome.tif'
registration_set=collection_parser(directory_path,pattern,'t','c',template_image)
    
    
    
    