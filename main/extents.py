
from typing import Literal

def validate_extent(extent, extent_format: Literal['Firms', 'CDS'] = 'FIRMS'):
    '''
    check input is valid set of FIRMS extents
    (min long, min lat, max long, max lat)

    written like test so will raise errors if there are issues
    there are probably better ways to handle it but eh
    '''

    if extent_format == 'CDS':
        extent = CDS_to_FIRMS(extent)

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

def FIRMS_to_CDS(extent):
        '''
        FIRMS to CDS form (north, west, south, east)
        '''

        ## note this is the same as rotating the array one to the right:
        ## [extent[3], extent[0], extent[1], extent[2]]

        CDS_extent = extent.copy()

        CDS_extent.insert(0, CDS_extent.pop())

        return CDS_extent

def CDS_to_FIRMS(extent):
    '''
    used exclusively for validation bcs i am lazy
    '''

    FIRMS_extent = extent.copy()

    FIRMS_extent.append(FIRMS_extent.pop(0))

    return FIRMS_extent

class Extent:

    def __init__(self, FIRMS_extent):
        '''
        extent in FIRMS form (min long, min lat, max long, max lat)
        '''

        validate_extent(FIRMS_extent)

        self.FIRMS = FIRMS_extent

    @property
    def CDS(self):
        '''
        in CDS form (north, west, south, east)
        '''

        return FIRMS_to_CDS(self.FIRMS)

## INTERESTING REGIONS

# small section around the alaska range
ALASKA_RANGE_EXTENT = Extent([-153.7, 61.3, -140.7, 65.0])

# basically all of the western half of washington and oregon
OREGON_COAST_RANGE_EXTENT = Extent([-126.0, 41.9, -116.438, 49.0])

# all of sumatra, and peninsular malaysia
INDO_MALAY_RANGE_EXTENT = Extent([94.9, -6.7, 106.6, 6.5]) 

# northern tip of Northern Territory and Queensland (AUS)
NORTH_AUSTRALIA_EXTENT = Extent([129.0, -16.9, 145.5, -10.1])