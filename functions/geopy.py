# Creating a function to get the city from the coordinates
geolocator = Nominatim(user_agent="geoapiExercises")
def city(coord):
    location = geolocator.reverse(coord, exactly_one=True)
    address = location.raw['address']
    city = address.get('city', '')
    return city

# Creating a function to get the country from the coordinates
def country(coord):
    location = geolocator.reverse(coord, exactly_one=True)
    address = location.raw['address']
    state = address.get('country', '')
    return state