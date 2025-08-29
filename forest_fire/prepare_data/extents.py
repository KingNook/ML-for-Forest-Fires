'''
Provides the `Extent` class which allows for easy conversions between FIRMS and CDS forms of coordinates, as well as pre-defined extents
'''

def FIRMS_to_CDS(extent):
        '''
        Converts from FIRMS form to CDS form. Note that this is the same as rotating the list one place to the right.

        Parameters
        ----------
        extent: list
            List of coordinates in FIRMS form: `(min long, min lat, max long, max lat)`

        Returns
        -------
        transformed_extent: list
            List of coordinates in CDS form: `(north, west, south, east)`
        '''

        CDS_extent = extent.copy()
        CDS_extent.insert(0, CDS_extent.pop())

        return CDS_extent

class Extent:
    '''
    An object that allows for easy conversion between CDS and FIRMS forms, for convenience's sake

    Parameters
    ----------
    extent: list or tuple
        The list of coordinates **in FIRMS form**, that is, `(min long, min lat, max long, max lat)`

    name: str, optional
        Name of the extent; if not provided, will be an empty string
    '''

    def __init__(self, extent, name = ''):

        self.FIRMS = extent
        self.CDS = FIRMS_to_CDS(extent)
        self.name = name

        self.validate()

    def validate(self):
        '''
        Checks whether input is a valid extent for data requests; raises an error if format is invalid
        '''

        extent = self.FIRMS

        assert type(extent) == list or type(extent) == tuple, 'Invalid type'

        assert len(extent) == 4, 'Invalid length'

        ## latitude checks -- between -90 and 90

        assert -90 <= extent[1] <= 90, 'Invalid minimum latitude'
        assert -90 <= extent[3] <= 90, f'Invalid maximum latitude, {extent[3] = }'

        ## longitude checks -- between -180 and 180

        assert -180 <= extent[0] <= 180, 'Invalid minimum longitude'
        assert -180 <= extent[2] <= 180, 'Invalid maximum longitude'

        ## nb don't need to check if min long < max long since if not, we assume this to mean go through antimeridian
        ## idk what happens if min lat > max lat

## ===================
## INTERESTING REGIONS
## ===================

# small section around the alaska range
## NB THIS IS TOO SMALL AND DOESN'T REALLY HAVE ANY FIRES
ALASKA_RANGE_EXTENT = Extent([-153.7, 61.3, -140.7, 65.0], name='alaska_range')

# basically all of the western half of washington and oregon
OREGON_COAST_RANGE_EXTENT = Extent([-126.0, 41.9, -116.438, 49.0], name='oregon_coast')

# all of sumatra, and peninsular malaysia
INDO_MALAY_RANGE_EXTENT = Extent([94.9, -6.7, 106.6, 6.5], name='indo_malay') 

# northern tip of Northern Territory and Queensland (AUS)
NORTH_AUSTRALIA_EXTENT = Extent([129.0, -16.9, 145.5, -10.1], name='north_aus')

# california (should all be land)
LA_FORESTS_EXTENT = Extent([-118.3,33.9,-116.5,34.7], name='la_forest')

# canada richardson fire
CANADA_RICHARDSON_EXTENT = Extent([-113.3, 55.7, -108.6, 59.1], name='canada_richardson')

# central / south-central africa
CENTRAL_AFRICA_EXTENT = Extent([10.1,-19.3,42,-2], name='central_africa')