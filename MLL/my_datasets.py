from torchvision import datasets
from pathlib import Path
from PIL import Image
from typing import Any, Callable, Optional, Tuple, Union

class MyMNIST(datasets.MNIST):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode="L").convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MyPCAM(datasets.PCAM):
    def __init__(self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(root, split, transform, target_transform, download)
        self.classes = ['negative', 'positive']
        self.class_to_idx = {'negative': 0, 'positive': 1}

class MyCountry211(datasets.Country211):
    def __init__(
            self,
            root: Union[str, Path],
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
        ) -> None:
            super().__init__(root, split, transform, target_transform, download)
            self.classes = self.country211_transfer(self.classes)
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def country211_transfer(self, raws):
        country_codes = {
            "AD": "Andorra",
            "AE": "United Arab Emirates",
            "AF": "Afghanistan",
            "AG": "Antigua and Barbuda",
            "AI": "Anguilla",
            "AL": "Albania",
            "AM": "Armenia",
            "AO": "Angola",
            "AQ": "Antarctica",
            "AR": "Argentina",
            "AS": "American Samoa",
            "AT": "Austria",
            "AU": "Australia",
            "AW": "Aruba",
            "AX": "Åland Islands",
            "AZ": "Azerbaijan",
            "BA": "Bosnia and Herzegovina",
            "BB": "Barbados",
            "BD": "Bangladesh",
            "BE": "Belgium",
            "BF": "Burkina Faso",
            "BG": "Bulgaria",
            "BH": "Bahrain",
            "BI": "Burundi",
            "BJ": "Benin",
            "BL": "Saint Barthélemy",
            "BM": "Bermuda",
            "BN": "Brunei Darussalam",
            "BO": "Bolivia, Plurinational State of",
            "BQ": "Bonaire, Sint Eustatius and Saba",
            "BR": "Brazil",
            "BS": "Bahamas",
            "BT": "Bhutan",
            "BV": "Bouvet Island",
            "BW": "Botswana",
            "BY": "Belarus",
            "BZ": "Belize",
            "CA": "Canada",
            "CC": "Cocos (Keeling) Islands",
            "CD": "Congo, Democratic Republic of the",
            "CF": "Central African Republic",
            "CG": "Congo",
            "CH": "Switzerland",
            "CI": "Côte d'Ivoire",
            "CK": "Cook Islands",
            "CL": "Chile",
            "CM": "Cameroon",
            "CN": "China",
            "CO": "Colombia",
            "CR": "Costa Rica",
            "CU": "Cuba",
            "CV": "Cabo Verde",
            "CW": "Curaçao",
            "CX": "Christmas Island",
            "CY": "Cyprus",
            "CZ": "Czechia",
            "DE": "Germany",
            "DJ": "Djibouti",
            "DK": "Denmark",
            "DM": "Dominica",
            "DO": "Dominican Republic",
            "DZ": "Algeria",
            "EC": "Ecuador",
            "EE": "Estonia",
            "EG": "Egypt",
            "EH": "Western Sahara",
            "ER": "Eritrea",
            "ES": "Spain",
            "ET": "Ethiopia",
            "FI": "Finland",
            "FJ": "Fiji",
            "FK": "Falkland Islands (Malvinas)",
            "FM": "Micronesia, Federated States of",
            "FO": "Faroe Islands",
            "FR": "France",
            "GA": "Gabon",
            "GB": "United Kingdom",
            "GD": "Grenada",
            "GE": "Georgia",
            "GF": "French Guiana",
            "GG": "Guernsey",
            "GH": "Ghana",
            "GI": "Gibraltar",
            "GL": "Greenland",
            "GM": "Gambia",
            "GN": "Guinea",
            "GP": "Guadeloupe",
            "GQ": "Equatorial Guinea",
            "GR": "Greece",
            "GS": "South Georgia and the South Sandwich Islands",
            "GT": "Guatemala",
            "GU": "Guam",
            "GW": "Guinea-Bissau",
            "GY": "Guyana",
            "HK": "Hong Kong",
            "HM": "Heard Island and McDonald Islands",
            "HN": "Honduras",
            "HR": "Croatia",
            "HT": "Haiti",
            "HU": "Hungary",
            "ID": "Indonesia",
            "IE": "Ireland",
            "IL": "Israel",
            "IM": "Isle of Man",
            "IN": "India",
            "IO": "British Indian Ocean Territory",
            "IQ": "Iraq",
            "IR": "Iran, Islamic Republic of",
            "IS": "Iceland",
            "IT": "Italy",
            "JE": "Jersey",
            "JM": "Jamaica",
            "JO": "Jordan",
            "JP": "Japan",
            "KE": "Kenya",
            "KG": "Kyrgyzstan",
            "KH": "Cambodia",
            "KI": "Kiribati",
            "KM": "Comoros",
            "KN": "Saint Kitts and Nevis",
            "KP": "Korea, Democratic People's Republic of",
            "KR": "Korea, Republic of",
            "KW": "Kuwait",
            "KY": "Cayman Islands",
            "KZ": "Kazakhstan",
            "LA": "Lao People's Democratic Republic",
            "LB": "Lebanon",
            "LC": "Saint Lucia",
            "LI": "Liechtenstein",
            "LK": "Sri Lanka",
            "LR": "Liberia",
            "LS": "Lesotho",
            "LT": "Lithuania",
            "LU": "Luxembourg",
            "LV": "Latvia",
            "LY": "Libya",
            "MA": "Morocco",
            "MC": "Monaco",
            "MD": "Moldova, Republic of",
            "ME": "Montenegro",
            "MF": "Saint Martin (French part)",
            "MG": "Madagascar",
            "MH": "Marshall Islands",
            "MK": "North Macedonia",
            "ML": "Mali",
            "MM": "Myanmar",
            "MN": "Mongolia",
            "MO": "Macao",
            "MP": "Northern Mariana Islands",
            "MQ": "Martinique",
            "MR": "Mauritania",
            "MS": "Montserrat",
            "MT": "Malta",
            "MU": "Mauritius",
            "MV": "Maldives",
            "MW": "Malawi",
            "MX": "Mexico",
            "MY": "Malaysia",
            "MZ": "Mozambique",
            "NA": "Namibia",
            "NC": "New Caledonia",
            "NE": "Niger",
            "NF": "Norfolk Island",
            "NG": "Nigeria",
            "NI": "Nicaragua",
            "NL": "Netherlands",
            "NO": "Norway",
            "NP": "Nepal",
            "NR": "Nauru",
            "NU": "Niue",
            "NZ": "New Zealand",
            "OM": "Oman",
            "PA": "Panama",
            "PE": "Peru",
            "PF": "French Polynesia",
            "PG": "Papua New Guinea",
            "PH": "Philippines",
            "PK": "Pakistan",
            "PL": "Poland",
            "PM": "Saint Pierre and Miquelon",
            "PN": "Pitcairn",
            "PR": "Puerto Rico",
            "PS": "Palestine, State of",
            "PT": "Portugal",
            "PW": "Palau",
            "PY": "Paraguay",
            "QA": "Qatar",
            "RE": "Réunion",
            "RO": "Romania",
            "RS": "Serbia",
            "RU": "Russian Federation",
            "RW": "Rwanda",
            "SA": "Saudi Arabia",
            "SB": "Solomon Islands",
            "SC": "Seychelles",
            "SD": "Sudan",
            "SE": "Sweden",
            "SG": "Singapore",
            "SH": "Saint Helena, Ascension and Tristan da Cunha",
            "SI": "Slovenia",
            "SJ": "Svalbard and Jan Mayen",
            "SK": "Slovakia",
            "SL": "Sierra Leone",
            "SM": "San Marino",
            "SN": "Senegal",
            "SO": "Somalia",
            "SR": "Suriname",
            "SS": "South Sudan",
            "ST": "Sao Tome and Principe",
            "SV": "El Salvador",
            "SX": "Sint Maarten (Dutch part)",
            "SY": "Syrian Arab Republic",
            "SZ": "Eswatini",
            "TC": "Turks and Caicos Islands",
            "TD": "Chad",
            "TF": "French Southern Territories",
            "TG": "Togo",
            "TH": "Thailand",
            "TJ": "Tajikistan",
            "TK": "Tokelau",
            "TL": "Timor-Leste",
            "TM": "Turkmenistan",
            "TN": "Tunisia",
            "TO": "Tonga",
            "TR": "Türkiye",
            "TT": "Trinidad and Tobago",
            "TV": "Tuvalu",
            "TW": "Taiwan, Province of China",
            "TZ": "Tanzania, United Republic of",
            "UA": "Ukraine",
            "UG": "Uganda",
            "UM": "United States Minor Outlying Islands",
            "US": "United States of America",
            "UY": "Uruguay",
            "UZ": "Uzbekistan",
            "VA": "Holy See",
            "VC": "Saint Vincent and the Grenadines",
            "VE": "Venezuela, Bolivarian Republic of",
            "VG": "Virgin Islands (British)",
            "VI": "Virgin Islands (U.S.)",
            "VN": "Viet Nam",
            "VU": "Vanuatu",
            "WF": "Wallis and Futuna",
            "WS": "Samoa",
            "YE": "Yemen",
            "YT": "Mayotte",
            "ZA": "South Africa",
            "ZM": "Zambia",
            "ZW": "Zimbabwe",

            "XK": "Kosovo"

        }

        return [country_codes[raw] for raw in raws]

def build_all_dataset(dataset_list, is_train, transform=None):
    all_datasets = dict()
    for dataset_name in dataset_list:
        ds = build_dataset(dataset_name, dataset_list[dataset_name]['root'], is_train, transform)
        all_datasets.update({dataset_name: ds})
    return all_datasets

def build_dataset(dataset_name, dataset_path, is_train=False, transform=None):
    if is_train:
        spilt = 'train'
        spilt2 = 'train'
    else:
        spilt = 'valid'
        spilt2 = 'val'
        spilt3 = 'test'

        if dataset_name == 'CIFAR100':
            dataset = datasets.CIFAR100(dataset_path, train=is_train, transform=transform)
        elif dataset_name == 'Country211':
            dataset = MyCountry211(dataset_path, split=spilt, transform=transform)
        elif dataset_name == 'DTD':
            dataset = datasets.DTD(dataset_path, split=spilt2, transform=transform)
        elif dataset_name == 'Food101':
            dataset = datasets.Food101(dataset_path, split=spilt3, transform=transform)
        elif dataset_name == 'MNIST':
            dataset = MyMNIST(dataset_path, train=is_train, transform=transform)
        elif dataset_name == 'OxfordIIITPet':
            dataset = datasets.OxfordIIITPet(dataset_path, split=spilt3, transform=transform)
        elif dataset_name == 'PCAM':
            dataset = MyPCAM(dataset_path, split=spilt3, transform=transform)
        elif dataset_name == 'RenderedSST2':
            dataset = datasets.RenderedSST2(dataset_path,split=spilt3,transform=transform)
        elif dataset_name == 'StanfordCars':
            dataset = datasets.StanfordCars(dataset_path, split=spilt3, transform=transform)
        elif dataset_name == 'STL10':
                dataset = datasets.STL10(dataset_path, split=spilt3, transform=transform)
        elif dataset_name == 'ImageNet':
            dataset = datasets.ImageNet(dataset_path, split=spilt2, transform=transform)
        else:
            dataset = datasets.ImageFolder(dataset_path, transform=transform)
    return dataset



