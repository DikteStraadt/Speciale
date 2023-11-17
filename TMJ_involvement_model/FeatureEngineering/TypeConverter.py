from pandas import CategoricalDtype

class ConvertToCategories:

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):

        column_categories = {
            'activity': CategoricalDtype(ordered=False),
            'involvementstatus': CategoricalDtype(ordered=False),
            'drug': CategoricalDtype(ordered=False),
            'swallenright': CategoricalDtype(ordered=True),
            'swallenleft': CategoricalDtype(ordered=True),
            'clickright': CategoricalDtype(ordered=True),
            'clickleft': CategoricalDtype(ordered=True),
            'lockright': CategoricalDtype(ordered=True),
            'lockleft': CategoricalDtype(ordered=True),
            'crepitationright': CategoricalDtype(ordered=True),
            'crepitationleft': CategoricalDtype(ordered=True),
            'painright': CategoricalDtype(ordered=True),
            'painleft': CategoricalDtype(ordered=True),
            'painmoveright': CategoricalDtype(ordered=True),
            'painmoveleft': CategoricalDtype(ordered=True),
            'morningstiffness': CategoricalDtype(ordered=True),
            'muscularpainright': CategoricalDtype(ordered=True),
            'muscularpainleft': CategoricalDtype(ordered=True),
            'headache': CategoricalDtype(ordered=True),
            'bruxism': CategoricalDtype(ordered=True),
            'chewingfunction': CategoricalDtype(ordered=True),
            'openingfunction': CategoricalDtype(ordered=True),
            'neckpain': CategoricalDtype(ordered=True),
            'neckstiffness': CategoricalDtype(ordered=True),
            'asypupilline': CategoricalDtype(ordered=True),
            'asybasis': CategoricalDtype(ordered=True),
            'asyoccl': CategoricalDtype(ordered=True),
            'asymenton': CategoricalDtype(ordered=True),
            'asyupmid': CategoricalDtype(ordered=True),
            'asylowmi': CategoricalDtype(ordered=True),
            'profile': CategoricalDtype(ordered=False),
            'lowerface': CategoricalDtype(ordered=True),
            'retrognathism': CategoricalDtype(ordered=True),
            'lips': CategoricalDtype(ordered=False),
            'respiration': CategoricalDtype(ordered=True),
            'tongue': CategoricalDtype(ordered=True),
            'dualbite': CategoricalDtype(ordered=True),
            'neckpalpation': CategoricalDtype(ordered=True),
            'backbending': CategoricalDtype(ordered=True),
            'forwardbending': CategoricalDtype(ordered=True),
            'rotationright': CategoricalDtype(ordered=True),
            'rotationleft': CategoricalDtype(ordered=True),
            'swallenjointright': CategoricalDtype(ordered=True),
            'swallenjointleft': CategoricalDtype(ordered=True),
            'laterpalpright': CategoricalDtype(ordered=True),
            'laterpalpleft': CategoricalDtype(ordered=True),
            'postpalpright': CategoricalDtype(ordered=True),
            'postpalpleft': CategoricalDtype(ordered=True),
            'clickopeningright': CategoricalDtype(ordered=True),
            'clickopeningleft': CategoricalDtype(ordered=True),
            'clickclosingright': CategoricalDtype(ordered=True),
            'clickclosingleft': CategoricalDtype(ordered=True),
            'clicklaterotrusionright': CategoricalDtype(ordered=True),
            'clicklaterotrusionleft': CategoricalDtype(ordered=True),
            'clickprotrusionright': CategoricalDtype(ordered=True),
            'clickprotrusionleft': CategoricalDtype(ordered=True),
            'translationright': CategoricalDtype(ordered=True),
            'translationleft': CategoricalDtype(ordered=True),
            'hypermobilityright': CategoricalDtype(ordered=True),
            'hypermobilityleft': CategoricalDtype(ordered=True),
            'Krepitationright': CategoricalDtype(ordered=True),
            'Krepitationleft': CategoricalDtype(ordered=True),
            'masseterright': CategoricalDtype(ordered=True),
            'masseterleft': CategoricalDtype(ordered=True),
            'temporalisright': CategoricalDtype(ordered=True),
            'temporalisleft': CategoricalDtype(ordered=True),
            'ptextright': CategoricalDtype(ordered=True),
            'ptextleft': CategoricalDtype(ordered=True),
            'ptintright': CategoricalDtype(ordered=True),
            'ptintleft': CategoricalDtype(ordered=True),
            'tempsenright': CategoricalDtype(ordered=True),
            'tempsenleft': CategoricalDtype(ordered=True),
            'sternoright': CategoricalDtype(ordered=True),
            'sternoleft': CategoricalDtype(ordered=True),
            'stylomandibularligamentright': CategoricalDtype(ordered=True),
            'stylomandibularligamentleft': CategoricalDtype(ordered=True),
            'asymmetrymasseterright': CategoricalDtype(ordered=True),
            'asymmetrymasseterleft': CategoricalDtype(ordered=True),
            'spacerelationship': CategoricalDtype(ordered=False),
            'abrasion': CategoricalDtype(ordered=True),
            'aplasia': CategoricalDtype(ordered=True),
            'sagittalrelationright': CategoricalDtype(ordered=False),
            'sagitalrelationleft': CategoricalDtype(ordered=False),
            'overjet': CategoricalDtype(ordered=True),
            'overbite': CategoricalDtype(ordered=True),
            'openbite': CategoricalDtype(ordered=True),
            'deepbite': CategoricalDtype(ordered=True),
            'transversal': CategoricalDtype(ordered=False),
        }

        data = data.astype(column_categories)

        non_categorical_columns = ['ID', 'ageatvisitation', 'difftdate', 'openingmm', 'opening', 'protrusionmm', 'protrusion', 'laterotrusionrightmm', 'laterotrusionleftmm']
        categorical_columns = [col for col in data.columns if col not in non_categorical_columns]
        for col in categorical_columns:
            data[col] = data[col].astype('category').cat.codes

        return data

