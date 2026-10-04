# Dataset attribution and source notice

The source package contains code and provenance metadata only. It does not
redistribute EPIC video, images, audio, or annotation CSVs. Dataset licensing is
separate from the repository's code; no license is invented or changed by this
patch.

EPIC-KITCHENS P02_05: Dima Damen and the EPIC-KITCHENS contributors. Original
EK55 media used with EK100 annotations, obtained from the
[official dataset source](https://epic-kitchens.github.io/2025#download).
License: [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).
No endorsement is implied. Original video bytes were retained unchanged locally.
Derived frames are sampled/resized, and audio is converted to mono 16 kHz;
model outputs, if produced, are separately identified as predictions.

## Dataset papers

- Dima Damen et al. *Scaling Egocentric Vision: The EPIC-KITCHENS Dataset.*
  European Conference on Computer Vision (ECCV), 2018.
  [Official paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Dima_Damen_Scaling_Egocentric_Vision_ECCV_2018_paper.html)
- Dima Damen et al. *Rescaling Egocentric Vision: Collection, Pipeline and
  Challenges for EPIC-KITCHENS-100.* International Journal of Computer Vision,
  130:33–55, 2022. [DOI](https://doi.org/10.1007/s11263-021-01531-2)

Official BibTeX and the complete author lists are provided on the
[dataset publication page](https://epic-kitchens.github.io/2025#download).

EPIC-SOUNDS reference labels: Jaesung Huh, Jacob Chalk, Evangelos Kazakos,
Dima Damen, Andrew Zisserman and contributors. See the
[official EPIC-SOUNDS source](https://github.com/epic-kitchens/epic-sounds-annotations)
for applicable attribution and publication details. These labels stay outside
pipeline observations and are not ASR ground truth.
