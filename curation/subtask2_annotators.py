import os
import pandas as pd
from tqdm import tqdm
from subtask2 import Subtask2Annotations, get_ref_df
# midfix = "s" 
midfix = "test_s"

if __name__ == "__main__":
    # Change per run: 
    samples = [1,2] #list(range(1,8+1))
    A_odd = ['ali','farhana']
    B_even = ['tommaso','onur','tadashi']
    root_ann_folder = r"D:\61 Challenges\2022_CASE_\WebAnno\reviewing_annotations\Subtask2\07. Round5\annotation"
    
    # Do not touch the remaining:
    ref_df = get_ref_df()
    metrics_df = pd.DataFrame()
    passed = 0
    for sub in tqdm(samples):

        # Parse annotations to CSV
        st2a = Subtask2Annotations(
            ref_df = ref_df,
            root_ann_folder = root_ann_folder, 
            folder_name = "subtask2_{0}{1:02d}.txt".format(midfix, sub),
            annotators = A_odd if sub%2!=0 else B_even,
            add_cleanedtext = False
            )
        st2a.parse()
        passed+=st2a.prepare_report(sub, split_by_annotator=True)
        
        # Agreement scores
        st2a.calculate_pico()
        df = st2a.format_metrics()
        metrics_df = pd.concat([metrics_df,df],axis=0)
        print(st2a.metrics)
        st2a.reset_metrics()

    metrics_df.to_csv(
        os.path.join(root_ann_folder, "agreement_all_{0}{1:02d}_s{2:02d}.csv".format(midfix, min(samples), sub)), 
        index=False, encoding='utf-8-sig'
        )
    print(f'Proportion of passed subsamples: {passed/len(samples)}')


    

