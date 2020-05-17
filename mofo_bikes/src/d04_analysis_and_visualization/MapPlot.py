import folium
import geopandas

def create_map(data, lat, lon, geo_data, nycha = True,):
    '''
    Creates a map ploting nycha buildings and dock station using the data.
    REQUIRES NYCHA json file as 'geo_data'. If you don't have the geospatial data set nycha to false
    Set the lat and lon to their respestive column name.
    '''
    Manhattan = [40.76, -74.00]
    m = folium.Map(location = Manhattan,
                   tiles = 'OpenStreetMap',
                   zoom_start = 12
                  )
    if nycha == True:
        m.choropleth(geo_data = geo_data)
    
    docker = data.loc[:,[lat,lon]]
    docker = docker.drop_duplicates()
    docker = docker.reset_index()
    
    for i in range(docker.shape[0]):
        folium.CircleMarker(location =[docker[lat][i], docker[lon][i] ],
                            radius = 3,
                            color = 'red'
                           ).add_to(m)
    return m