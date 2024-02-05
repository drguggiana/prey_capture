import numpy as np
import pandas as pd
import functions_bondjango as bd
import functions_data_handling as fdh

from pprint import pprint

df_list = []
for mouse in ['MM_221110_a', 'MM_221109_a', 'MM_220928_a', 'MM_220915_a',
              'MM_230518_b', 'MM_230705_b', 'MM_230706_a', 'MM_230706_b']:
    # get the search string and paths
    search_string = f"mouse:{mouse}"
    parsed_search = fdh.parse_search_string(search_string)
    file_infos = bd.query_database("analyzed_data", search_string)

    preprocessing_paths = np.array([el['slug'] for el in file_infos if ('_preproc' in el['slug']) and
                                   (parsed_search['mouse'].lower() in el['slug'])])
    preproc_slugs = np.array(["_".join(pp_path.split('_')[:-1]) for pp_path in preprocessing_paths])

    preproc_dates = [el['slug'][:10] for el in file_infos if ('_preproc' in el['slug']) and
                     (parsed_search['mouse'].lower() in el['slug'])]
    unique_preproc_dates, preproc_counts = np.unique(preproc_dates, return_counts=True)
    matched_preproc_dates = unique_preproc_dates[preproc_counts > 1]
    matched_preproc = np.in1d(preproc_dates, matched_preproc_dates)
    pprint(mouse)

    tcday_paths = np.array([el['slug'] for el in file_infos if ('_tcday' in el['slug']) and
                            (parsed_search['mouse'].lower() in el['slug'])])
    tcday_slugs = np.array(["_".join(tc_path.split('_')[:-1]) for tc_path in tcday_paths])

    processed_tcdays_bool = np.in1d(preproc_slugs, tcday_slugs)

    tccons_paths = np.array([el['slug'] for el in file_infos if ('_tcconsolidate' in el['slug']) and
                             (parsed_search['mouse'].lower() in el['slug'])])
    tccons_slugs = np.array(["_".join(tc_path.split('_')[:-1]) for tc_path in tccons_paths])
    tccons_date = np.array([el['slug'][:10] for el in file_infos if ('_tcconsolidate' in el['slug'])
                            and (parsed_search['mouse'].lower() in el['slug'])])

    processed_tccons_bool = np.in1d(preproc_dates, tccons_date)

    cols = ['date', 'preproc_files', 'is_matched', 'tcday_done', 'tcconsolidate_done']
    data = [preproc_dates, preproc_slugs, matched_preproc, processed_tcdays_bool, processed_tccons_bool]
    df = pd.DataFrame(data=dict(zip(cols, data)))
    df.insert(loc=0, column='mouse', value=mouse)
    df.sort_values('date', inplace=True)
    df_list.append(df)

data_df = pd.concat(df_list, axis=0)
data_df.to_csv(r'C:\Users\mmccann\Desktop\tc_status.csv')




