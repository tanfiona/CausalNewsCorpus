import os
import pandas as pd
from tqdm import tqdm
from subtask2 import Subtask2Annotations, get_ref_df
midfix = "s" 
midfix = "test_s"

if __name__ == "__main__":
    # Change per run: 
    samples = list(range(1,4+1))
    root_ann_folder = r"D:\61 Challenges\2022_CASE_\WebAnno\reviewing_annotations\Subtask2\15. Round13\annotation"
    
    # Do not touch the remaining:
    ref_df = get_ref_df()
    dmetrics_dict = {}
    metrics_df = pd.DataFrame()
    passed = 0
    for sub in tqdm(samples):

        # Parse annotations to CSV
        folder_name = "subtask2_{0}{1:02d}.txt".format(midfix, sub)
        annotators = [os.path.basename(f) for f in os.listdir(os.path.join(root_ann_folder,folder_name)) if str(f)[-5:]=='.json']
        annotators = [os.path.splitext(a)[0] for a in annotators if (a!='admin.json')]
        st2a = Subtask2Annotations(
            ref_df = ref_df,
            root_ann_folder = root_ann_folder, 
            folder_name = folder_name,
            annotators = annotators,
            add_cleanedtext = False
            )
        st2a.parse()
        # passed+=st2a.prepare_report(sub, split_by_annotator=True)
        
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


    

