import pandas as pd

featureList = [
    ['swallenright', 'swallenleft'],
    ['clickright', 'clickleft'],
    ['lockright', 'lockleft'],
    ['crepitationright','crepitationleft'],
    ['painright', 'painleft'],
    ['painmoveright', 'painmoveleft'],
    ['muscularpainright', 'muscularpainleft'],
    ['rotationright', 'rotationleft'],
    ['swallenjointright', 'swallenjointleft'],
    ['laterpalpright', 'laterpalpleft'],
    ['postpalpright', 'postpalpleft'],
    ['clickopeningright', 'clickopeningleft'],
    ['clickclosingright', 'clickclosingleft'],
    ['clicklaterotrusionright', 'clicklaterotrusionleft'],
    ['clickprotrusionright', 'clickprotrusionleft'],
    ['translationright', 'translationleft'],
    ['hypermobilityright', 'hypermobilityleft'],
    ['Krepitationright', 'Krepitationleft'],
    ['masseterright', 'masseterleft'],
    ['temporalisright', 'temporalisleft'],
    ['ptextright', 'ptextleft'],
    ['ptintright', 'ptintleft'],
    ['tempsenright', 'tempsenleft'],
    ['sternoright', 'sternoleft'],
    ['stylomandibularligamentright', 'stylomandibularligamentleft'],
    ['asymmetrymasseterright', 'asymmetrymasseterleft'],
    ['sagittalrelationright', 'sagitalrelationleft'],
]  # 54


def getHighestSeverity(right, left):
    mergedList = []
    for i in range(len(right)):
        highVal = max (right[i], left[i])
        mergedList.append(highVal)

    return mergedList


class MergeFeatures:
    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        new_df = data

        for featurepair in featureList:
            featureNameMerged = featurepair[0].removesuffix('right')
            mergedList = getHighestSeverity(new_df[featurepair[0]], new_df[featurepair[1]])
            new_df[featureNameMerged] = mergedList

            new_df.drop([featurepair[0], featurepair[1]], axis=1, inplace=True)

        print("Merging of features is done")

        return new_df


