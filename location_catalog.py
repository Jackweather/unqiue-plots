from __future__ import annotations

ENABLED_STATE_KEYS = (
    "new_york",
    "maryland",
    "new_jersey",
    "connecticut",
    "maine",
    "massachusetts",
    "vermont",
)

STATE_CITY_GROUPS = {
    "alabama": {
        "label": "Alabama",
        "model_domain": "conus",
        "cities": [("huntsville", "Huntsville", "North"), ("birmingham", "Birmingham", "North/Central"), ("montgomery", "Montgomery", "Central"), ("mobile", "Mobile", "Southwest")],
    },
    "alaska": {
        "label": "Alaska",
        "model_domain": None,
        "cities": [("utqiagvik", "Utqiaġvik", "North"), ("fairbanks", "Fairbanks", "Interior"), ("anchorage", "Anchorage", "South-central"), ("juneau", "Juneau", "Southeast")],
    },
    "arizona": {
        "label": "Arizona",
        "model_domain": "conus",
        "cities": [("kingman", "Kingman", "Northwest"), ("flagstaff", "Flagstaff", "North"), ("phoenix", "Phoenix", "Central/South"), ("tucson", "Tucson", "Southeast")],
    },
    "arkansas": {
        "label": "Arkansas",
        "model_domain": "conus",
        "cities": [("fayetteville", "Fayetteville", "Northwest"), ("fort_smith", "Fort Smith", "West"), ("little_rock", "Little Rock", "Central"), ("jonesboro", "Jonesboro", "Northeast")],
    },
    "california": {
        "label": "California",
        "model_domain": "conus",
        "cities": [("redding", "Redding", "North"), ("sacramento", "Sacramento", "North/Central"), ("los_angeles", "Los Angeles", "South"), ("san_diego", "San Diego", "Southwest")],
    },
    "colorado": {
        "label": "Colorado",
        "model_domain": "conus",
        "cities": [("grand_junction", "Grand Junction", "West"), ("fort_collins", "Fort Collins", "North"), ("denver", "Denver", "Central"), ("pueblo", "Pueblo", "Southeast")],
    },
    "connecticut": {
        "label": "Connecticut",
        "model_domain": "conus",
        "cities": [("danbury", "Danbury", "West"), ("hartford", "Hartford", "Central"), ("new_haven", "New Haven", "South/East"), ("norwich", "Norwich", "Southeast")],
    },
    "delaware": {
        "label": "Delaware",
        "model_domain": "conus",
        "cities": [("wilmington", "Wilmington", "North"), ("dover", "Dover", "Central"), ("georgetown", "Georgetown", "South"), ("rehoboth_beach", "Rehoboth Beach", "Southeast")],
    },
    "florida": {
        "label": "Florida",
        "model_domain": "conus",
        "cities": [("pensacola", "Pensacola", "Northwest"), ("jacksonville", "Jacksonville", "Northeast"), ("tampa", "Tampa", "West/Central"), ("miami", "Miami", "Southeast")],
    },
    "georgia": {
        "label": "Georgia",
        "model_domain": "conus",
        "cities": [("rome", "Rome", "Northwest"), ("augusta", "Augusta", "East"), ("atlanta", "Atlanta", "Central/North"), ("valdosta", "Valdosta", "South")],
    },
    "hawaii": {
        "label": "Hawaii",
        "model_domain": None,
        "cities": [("waimea", "Waimea", "Big Island"), ("hilo", "Hilo", "Big Island"), ("kahului", "Kahului", "Maui"), ("honolulu", "Honolulu", "Oʻahu")],
    },
    "idaho": {
        "label": "Idaho",
        "model_domain": "conus",
        "cities": [("lewiston", "Lewiston", "Northwest"), ("coeur_dalene", "Coeur d'Alene", "North"), ("boise", "Boise", "Southwest"), ("idaho_falls", "Idaho Falls", "Southeast")],
    },
    "illinois": {
        "label": "Illinois",
        "model_domain": "conus",
        "cities": [("rockford", "Rockford", "North"), ("chicago", "Chicago", "Northeast"), ("springfield", "Springfield", "Central"), ("carbondale", "Carbondale", "South")],
    },
    "indiana": {
        "label": "Indiana",
        "model_domain": "conus",
        "cities": [("gary", "Gary", "Northwest"), ("fort_wayne", "Fort Wayne", "Northeast"), ("indianapolis", "Indianapolis", "Central"), ("evansville", "Evansville", "Southwest")],
    },
    "iowa": {
        "label": "Iowa",
        "model_domain": "conus",
        "cities": [("sioux_city", "Sioux City", "Northwest"), ("dubuque", "Dubuque", "Northeast"), ("des_moines", "Des Moines", "Central"), ("council_bluffs", "Council Bluffs", "Southwest")],
    },
    "kansas": {
        "label": "Kansas",
        "model_domain": "conus",
        "cities": [("goodland", "Goodland", "Northwest"), ("kansas_city", "Kansas City", "Northeast"), ("wichita", "Wichita", "South/Central"), ("dodge_city", "Dodge City", "Southwest")],
    },
    "kentucky": {
        "label": "Kentucky",
        "model_domain": "conus",
        "cities": [("paducah", "Paducah", "West"), ("louisville", "Louisville", "North"), ("lexington", "Lexington", "Central/East"), ("pikeville", "Pikeville", "East")],
    },
    "louisiana": {
        "label": "Louisiana",
        "model_domain": "conus",
        "cities": [("shreveport", "Shreveport", "Northwest"), ("monroe", "Monroe", "Northeast"), ("baton_rouge", "Baton Rouge", "Central"), ("new_orleans", "New Orleans", "Southeast")],
    },
    "maine": {
        "label": "Maine",
        "model_domain": "conus",
        "cities": [("fort_kent", "Fort Kent", "North"), ("bangor", "Bangor", "Central/East"), ("portland", "Portland", "Southwest"), ("presque_isle", "Presque Isle", "Northeast")],
    },
    "maryland": {
        "label": "Maryland",
        "model_domain": "conus",
        "cities": [("cumberland", "Cumberland", "West"), ("hagerstown", "Hagerstown", "Northwest"), ("baltimore", "Baltimore", "Central/East"), ("salisbury", "Salisbury", "Eastern Shore/South")],
    },
    "massachusetts": {
        "label": "Massachusetts",
        "model_domain": "conus",
        "cities": [("pittsfield", "Pittsfield", "West"), ("worcester", "Worcester", "Central"), ("boston", "Boston", "East"), ("new_bedford", "New Bedford", "Southeast")],
    },
    "michigan": {
        "label": "Michigan",
        "model_domain": "conus",
        "cities": [("ironwood", "Ironwood", "Western Upper Peninsula"), ("sault_ste_marie", "Sault Ste. Marie", "Eastern UP"), ("grand_rapids", "Grand Rapids", "West Lower Peninsula"), ("detroit", "Detroit", "Southeast")],
    },
    "minnesota": {
        "label": "Minnesota",
        "model_domain": "conus",
        "cities": [("duluth", "Duluth", "Northeast"), ("moorhead", "Moorhead", "Northwest"), ("minneapolis", "Minneapolis", "Central/South"), ("rochester", "Rochester", "Southeast")],
    },
    "mississippi": {
        "label": "Mississippi",
        "model_domain": "conus",
        "cities": [("clarksdale", "Clarksdale", "Northwest"), ("meridian", "Meridian", "East"), ("jackson", "Jackson", "Central"), ("gulfport", "Gulfport", "South")],
    },
    "missouri": {
        "label": "Missouri",
        "model_domain": "conus",
        "cities": [("st_joseph", "St. Joseph", "Northwest"), ("st_louis", "St. Louis", "East"), ("kansas_city", "Kansas City", "West"), ("springfield", "Springfield", "Southwest")],
    },
    "montana": {
        "label": "Montana",
        "model_domain": "conus",
        "cities": [("kalispell", "Kalispell", "Northwest"), ("glasgow", "Glasgow", "Northeast"), ("missoula", "Missoula", "West"), ("billings", "Billings", "Southeast")],
    },
    "nebraska": {
        "label": "Nebraska",
        "model_domain": "conus",
        "cities": [("scottsbluff", "Scottsbluff", "West"), ("norfolk", "Norfolk", "Northeast"), ("omaha", "Omaha", "East"), ("lincoln", "Lincoln", "Southeast/Central")],
    },
    "nevada": {
        "label": "Nevada",
        "model_domain": "conus",
        "cities": [("reno", "Reno", "Northwest"), ("elko", "Elko", "Northeast"), ("las_vegas", "Las Vegas", "South"), ("pahrump", "Pahrump", "Southwest")],
    },
    "new_hampshire": {
        "label": "New Hampshire",
        "model_domain": "conus",
        "cities": [("berlin", "Berlin", "North"), ("concord", "Concord", "Central"), ("manchester", "Manchester", "South"), ("keene", "Keene", "Southwest")],
    },
    "new_jersey": {
        "label": "New Jersey",
        "model_domain": "conus",
        "cities": [("sussex", "Sussex", "Northwest"), ("newark", "Newark", "Northeast"), ("trenton", "Trenton", "Central"), ("cape_may", "Cape May", "South")],
    },
    "new_mexico": {
        "label": "New Mexico",
        "model_domain": "conus",
        "cities": [("farmington", "Farmington", "Northwest"), ("santa_fe", "Santa Fe", "North/Central"), ("albuquerque", "Albuquerque", "Central"), ("las_cruces", "Las Cruces", "South")],
    },
    "new_york": {
        "label": "New York",
        "model_domain": "conus",
        "cities": [("buffalo", "Buffalo", "West"), ("syracuse", "Syracuse", "Central"), ("albany", "Albany", "East"), ("new_york_city", "New York City", "Southeast")],
    },
    "north_carolina": {
        "label": "North Carolina",
        "model_domain": "conus",
        "cities": [("asheville", "Asheville", "West"), ("greensboro", "Greensboro", "Central"), ("raleigh", "Raleigh", "East/Central"), ("wilmington", "Wilmington", "Southeast")],
    },
    "north_dakota": {
        "label": "North Dakota",
        "model_domain": "conus",
        "cities": [("williston", "Williston", "Northwest"), ("grand_forks", "Grand Forks", "Northeast"), ("bismarck", "Bismarck", "Central"), ("fargo", "Fargo", "Southeast")],
    },
    "ohio": {
        "label": "Ohio",
        "model_domain": "conus",
        "cities": [("toledo", "Toledo", "Northwest"), ("cleveland", "Cleveland", "Northeast"), ("columbus", "Columbus", "Central"), ("cincinnati", "Cincinnati", "Southwest")],
    },
    "oklahoma": {
        "label": "Oklahoma",
        "model_domain": "conus",
        "cities": [("guymon", "Guymon", "Northwest"), ("tulsa", "Tulsa", "Northeast"), ("oklahoma_city", "Oklahoma City", "Central"), ("lawton", "Lawton", "Southwest")],
    },
    "oregon": {
        "label": "Oregon",
        "model_domain": "conus",
        "cities": [("astoria", "Astoria", "Northwest"), ("pendleton", "Pendleton", "Northeast"), ("portland", "Portland", "Northwest/Central"), ("medford", "Medford", "Southwest")],
    },
    "pennsylvania": {
        "label": "Pennsylvania",
        "model_domain": "conus",
        "cities": [("erie", "Erie", "Northwest"), ("scranton", "Scranton", "Northeast"), ("pittsburgh", "Pittsburgh", "Southwest"), ("philadelphia", "Philadelphia", "Southeast")],
    },
    "rhode_island": {
        "label": "Rhode Island",
        "model_domain": "conus",
        "cities": [("westerly", "Westerly", "Southwest"), ("providence", "Providence", "North/Central"), ("newport", "Newport", "Southeast"), ("woonsocket", "Woonsocket", "Northwest")],
    },
    "south_carolina": {
        "label": "South Carolina",
        "model_domain": "conus",
        "cities": [("greenville", "Greenville", "Northwest"), ("columbia", "Columbia", "Central"), ("charleston", "Charleston", "Southeast"), ("myrtle_beach", "Myrtle Beach", "Northeast")],
    },
    "south_dakota": {
        "label": "South Dakota",
        "model_domain": "conus",
        "cities": [("rapid_city", "Rapid City", "West"), ("pierre", "Pierre", "Central"), ("aberdeen", "Aberdeen", "Northeast"), ("sioux_falls", "Sioux Falls", "Southeast")],
    },
    "tennessee": {
        "label": "Tennessee",
        "model_domain": "conus",
        "cities": [("memphis", "Memphis", "West"), ("nashville", "Nashville", "Central"), ("knoxville", "Knoxville", "East"), ("chattanooga", "Chattanooga", "Southeast")],
    },
    "texas": {
        "label": "Texas",
        "model_domain": "conus",
        "cities": [("el_paso", "El Paso", "Far West"), ("amarillo", "Amarillo", "Northwest"), ("houston", "Houston", "Southeast"), ("san_antonio", "San Antonio", "South/Central")],
    },
    "utah": {
        "label": "Utah",
        "model_domain": "conus",
        "cities": [("logan", "Logan", "North"), ("salt_lake_city", "Salt Lake City", "North/Central"), ("moab", "Moab", "Southeast"), ("st_george", "St. George", "Southwest")],
    },
    "vermont": {
        "label": "Vermont",
        "model_domain": "conus",
        "cities": [("st_albans", "St. Albans", "Northwest"), ("newport", "Newport", "Northeast"), ("burlington", "Burlington", "West/North"), ("brattleboro", "Brattleboro", "Southeast")],
    },
    "virginia": {
        "label": "Virginia",
        "model_domain": "conus",
        "cities": [("winchester", "Winchester", "Northwest"), ("richmond", "Richmond", "Central"), ("norfolk", "Norfolk", "Southeast"), ("bristol", "Bristol", "Southwest")],
    },
    "washington": {
        "label": "Washington",
        "model_domain": "conus",
        "cities": [("bellingham", "Bellingham", "Northwest"), ("spokane", "Spokane", "Northeast"), ("seattle", "Seattle", "West/Central"), ("walla_walla", "Walla Walla", "Southeast")],
    },
    "west_virginia": {
        "label": "West Virginia",
        "model_domain": "conus",
        "cities": [("wheeling", "Wheeling", "North"), ("morgantown", "Morgantown", "North/Central"), ("charleston", "Charleston", "Central"), ("bluefield", "Bluefield", "South")],
    },
    "wisconsin": {
        "label": "Wisconsin",
        "model_domain": "conus",
        "cities": [("superior", "Superior", "Northwest"), ("green_bay", "Green Bay", "Northeast"), ("milwaukee", "Milwaukee", "Southeast"), ("la_crosse", "La Crosse", "Southwest")],
    },
    "wyoming": {
        "label": "Wyoming",
        "model_domain": "conus",
        "cities": [("cody", "Cody", "Northwest"), ("sheridan", "Sheridan", "Northeast"), ("cheyenne", "Cheyenne", "Southeast"), ("rock_springs", "Rock Springs", "Southwest")],
    },
}


def get_enabled_state_groups() -> dict[str, dict[str, str | None | list[tuple[str, str, str]]]]:
    return {
        state_key: STATE_CITY_GROUPS[state_key]
        for state_key in ENABLED_STATE_KEYS
        if state_key in STATE_CITY_GROUPS
    }


def get_state_options() -> list[dict[str, str | bool | None]]:
    return [
        {
            "key": state_key,
            "label": state_data["label"],
            "model_domain": state_data["model_domain"],
            "supported": state_data["model_domain"] is not None,
        }
        for state_key, state_data in get_enabled_state_groups().items()
    ]


def iter_location_specs(max_locations_per_state: int | None = None) -> list[dict[str, str | None]]:
    locations: list[dict[str, str | None]] = []
    for state_key, state_data in get_enabled_state_groups().items():
        cities = state_data["cities"]
        if max_locations_per_state is not None and max_locations_per_state > 0:
            cities = cities[:max_locations_per_state]
        for city_key, city_label, region_label in cities:
            locations.append(
                {
                    "location_key": f"{state_key}__{city_key}",
                    "state_key": state_key,
                    "state_label": state_data["label"],
                    "city_key": city_key,
                    "city_label": city_label,
                    "region_label": region_label,
                    "model_domain": state_data["model_domain"],
                }
            )
    return locations