import os
import pandas as pd
from tqdm import tqdm
from subtask2 import Subtask2Annotations, get_ref_df
from itertools import combinations


if __name__ == "__main__":
    # Change per run: 
    samples = [1,7] #list(range(1,8+1))
    A_odd = ['ali','farhana']
    B_even = ['tommaso','onur','tadashi']
    root_ann_folder = r"D:\61 Challenges\2022_CASE_\WebAnno\reviewing_annotations\Subtask2\06. Round4\annotation"
    
    # Do not touch the remaining:
    ref_df = get_ref_df()
    passed = 0
    for sub in tqdm(samples):

        # Parse annotations to CSV
        st2a = Subtask2Annotations(
            ref_df = ref_df,
            root_ann_folder = root_ann_folder, 
            folder_name = "subtask2_s{:02d}.txt".format(sub),
            annotators = A_odd if sub%2!=0 else B_even,
            add_cleanedtext = False
            )
        st2a.parse()
        passed+=st2a.prepare_report(sub, split_by_annotator=True)
        
        # Agreement scores
        st2a.calculate_pico()
        print(st2a.metrics)
        st2a.reset_metrics()

    print(f'Proportion of passed subsamples: {passed/len(samples)}')


    

