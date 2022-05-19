import pandas as pd
import numpy as np
import random
import sklearn.model_selection as mod


def pad_latents(input_list, target_size):
    """Get the latents from a preprocessing file and add them to the main dataframe with the correct padding"""

    # allocate an output list
    output_list = []
    # for all the elements in the input list
    for el in input_list:
        # determine the delta size for padding
        delta_frames = target_size - el.shape[0]
        # pad latents due to the VAME calculation window
        latent_padding = pd.DataFrame(np.zeros((int(delta_frames / 2), len(el.columns))) * np.nan,
                                      columns=el.columns)
        # motif_padding = pd.DataFrame(np.zeros((int(delta_frames / 2), len(motifs.columns))) * np.nan,
        #                              columns=motifs.columns)
        # pad them with nan at the edges (due to VAME excluding the edges
        latents = pd.concat([latent_padding, el, latent_padding], axis=0).reset_index(drop=True)
        # add to the output list
        output_list.append(latents)
        # motifs = pd.concat([motif_padding, motifs, motif_padding], axis=0).reset_index(drop=True)

    return output_list

