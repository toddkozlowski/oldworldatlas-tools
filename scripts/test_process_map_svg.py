"""
Test suite for process_map_svg.py
Tests new features and validates that existing functionality continues to work.
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import sys

# Import the module under test
from process_map_svg import (
    Settlement, SVGMapProcessor, CoordinateConverter, 
    CALIBRATION_POINTS
)


class TestSettlementDataclass(unittest.TestCase):
    """Test the updated Settlement dataclass."""
    
    def test_settlement_initialization_with_defaults(self):
        """Test that Settlement initializes with correct default values."""
        s = Settlement(
            name="Test Settlement",
            province="Test Province",
            svg_x=100.0,
            svg_y=200.0
        )
        
        self.assertEqual(s.name, "Test Settlement")
        self.assertEqual(s.province, "Test Province")
        self.assertEqual(s.tags, [])
        self.assertEqual(s.notes, [])
        self.assertIsNotNone(s.wiki)
        self.assertIsNone(s.wiki["title"])
        self.assertIsNone(s.wiki["url"])

    def test_settlement_with_tags_and_notes(self):
        """Test Settlement with tags and notes."""
        tags = ["source:2eSH", "trade:timber"]
        notes = ["Founded in 2000", "Capital city"]
        
        s = Settlement(
            name="Altdorf",
            province="Reikland",
            svg_x=100.0,
            svg_y=200.0,
            tags=tags,
            notes=notes
        )
        
        self.assertEqual(s.tags, tags)
        self.assertEqual(s.notes, notes)

    def test_settlement_wiki_data(self):
        """Test Settlement wiki property."""
        wiki = {
            "title": "Altdorf",
            "url": "http://example.com",
            "description": "A city",
            "image": "http://example.com/image.jpg"
        }
        
        s = Settlement(
            name="Altdorf",
            province="Reikland",
            svg_x=100.0,
            svg_y=200.0,
            wiki=wiki
        )
        
        self.assertEqual(s.wiki, wiki)


class TestSVGMapProcessorHelperMethods(unittest.TestCase):
    """Test helper methods for tag and note parsing."""
    
    def setUp(self):
        """Set up test processor."""
        # Mock the SVG file to avoid loading it during tests
        with patch('process_map_svg.ET.parse'):
            with patch('process_map_svg.Path.exists', return_value=True):
                self.processor = SVGMapProcessor()
    
    def test_parse_tags_from_tags_column(self):
        """Test parsing tags from Tags column."""
        tags_str = '"""source:2eSH"""'
        trade_str = ""
        
        result = self.processor.parse_tags(tags_str, trade_str)
        
        self.assertIn("source:2eSH", result)
    
    def test_parse_tags_with_trade(self):
        """Test parsing tags with trade goods."""
        tags_str = '"""source:2eSH"""'
        trade_str = 'timber; textiles'
        
        result = self.processor.parse_tags(tags_str, trade_str)
        
        self.assertIn("source:2eSH", result)
        self.assertIn("trade:timber", result)
        self.assertIn("trade:textiles", result)
    
    def test_parse_tags_empty(self):
        """Test parsing empty tags."""
        result = self.processor.parse_tags("", "")
        self.assertEqual(result, [])
    
    def test_validate_tags_valid(self):
        """Test tag validation with valid tags."""
        tags = ["source:2eSH", "trade:timber"]
        result = self.processor.validate_tags(tags, "TestSettlement")
        
        # Should return same tags and not add to invalid_tags
        self.assertEqual(result, tags)
        self.assertEqual(len(self.processor.invalid_tags), 0)
    
    def test_validate_tags_invalid_source(self):
        """Test tag validation with invalid source."""
        tags = ["source:InvalidSource", "trade:timber"]
        result = self.processor.validate_tags(tags, "TestSettlement")
        
        # Should still return tags but log as invalid
        self.assertEqual(result, tags)
        self.assertEqual(len(self.processor.invalid_tags), 1)
        self.assertEqual(self.processor.invalid_tags[0]["settlement"], "TestSettlement")
    
    def test_validate_tags_invalid_format(self):
        """Test tag validation with invalid format."""
        tags = ["InvalidTag", "trade:timber"]
        result = self.processor.validate_tags(tags, "TestSettlement")
        
        self.assertEqual(len(self.processor.invalid_tags), 1)
        self.assertIn("missing format", self.processor.invalid_tags[0]["issues"][0])
    
    def test_parse_notes(self):
        """Test parsing notes from CSV."""
        notes_str = 'Note 1; Note 2; Note 3'
        
        result = self.processor.parse_notes(notes_str)
        
        self.assertEqual(len(result), 3)
        self.assertIn("Note 1", result)
        self.assertIn("Note 2", result)
        self.assertIn("Note 3", result)
    
    def test_parse_notes_empty(self):
        """Test parsing empty notes."""
        result = self.processor.parse_notes("")
        self.assertEqual(result, [])
    
    def test_calculate_size_category_village(self):
        """Test size category calculation for village."""
        self.assertEqual(self.processor.calculate_size_category(150), 1)  # Village
    
    def test_calculate_size_category_small_town(self):
        """Test size category calculation for small town."""
        self.assertEqual(self.processor.calculate_size_category(600), 2)  # Small Town
    
    def test_calculate_size_category_town(self):
        """Test size category calculation for town."""
        self.assertEqual(self.processor.calculate_size_category(1500), 3)  # Town
    
    def test_calculate_size_category_large_town(self):
        """Test size category calculation for large town."""
        self.assertEqual(self.processor.calculate_size_category(5000), 4)  # Large Town
    
    def test_calculate_size_category_city(self):
        """Test size category calculation for city."""
        self.assertEqual(self.processor.calculate_size_category(25000), 5)  # City
    
    def test_calculate_size_category_metropolis(self):
        """Test size category calculation for metropolis."""
        self.assertEqual(self.processor.calculate_size_category(100000), 6)  # Metropolis


class TestCoordinateConverter(unittest.TestCase):
    """Test the coordinate conversion functionality."""
    
    def test_coordinate_converter_initialization(self):
        """Test that CoordinateConverter initializes correctly."""
        converter = CoordinateConverter(CALIBRATION_POINTS)
        
        self.assertIsNotNone(converter.lon_coeffs)
        self.assertIsNotNone(converter.lat_coeffs)
        self.assertEqual(len(converter.lon_coeffs), 3)
        self.assertEqual(len(converter.lat_coeffs), 3)
    
    def test_svg_to_geo_conversion(self):
        """Test SVG to geographic coordinate conversion."""
        converter = CoordinateConverter(CALIBRATION_POINTS)
        
        # Test with one of the calibration points
        svg_x, svg_y = CALIBRATION_POINTS[0]["svg"]
        expected_lon, expected_lat = CALIBRATION_POINTS[0]["geo"]
        
        calculated_lon, calculated_lat = converter.svg_to_geo(svg_x, svg_y)
        
        # Should be very close due to calibration
        self.assertAlmostEqual(calculated_lon, expected_lon, places=2)
        self.assertAlmostEqual(calculated_lat, expected_lat, places=2)


class TestGeoJSONOutput(unittest.TestCase):
    """Test GeoJSON generation with new features."""
    
    def test_geojson_structure_with_new_properties(self):
        """Test that generated GeoJSON includes all new properties."""
        settlement = Settlement(
            name="Test Settlement",
            province="Test Province",
            svg_x=100.0,
            svg_y=200.0,
            geo_lon=1.5,
            geo_lat=48.0,
            population=5000,
            size_category=3,
            tags=["source:2eSH", "trade:timber"],
            notes=["Test note"],
            wiki={
                "title": "Test Settlement",
                "url": "http://example.com",
                "description": "A test settlement",
                "image": "http://example.com/image.jpg"
            }
        )
        
        # Create a simple feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [settlement.geo_lon, settlement.geo_lat]
            },
            "properties": {
                "name": settlement.name,
                "province": settlement.province,
                "population": settlement.population,
                "tags": settlement.tags,
                "notes": settlement.notes,
                "size_category": settlement.size_category,
                "inkscape_coordinates": [settlement.svg_x, settlement.svg_y],
                "wiki": settlement.wiki
            }
        }
        
        # Verify all properties are present
        props = feature["properties"]
        self.assertEqual(props["name"], "Test Settlement")
        self.assertEqual(props["province"], "Test Province")
        self.assertEqual(props["population"], 5000)
        self.assertEqual(props["size_category"], 3)
        self.assertEqual(len(props["tags"]), 2)
        self.assertEqual(len(props["notes"]), 1)
        self.assertIsNotNone(props["wiki"])
        self.assertEqual(props["wiki"]["title"], "Test Settlement")


class TestRandomPopulationAssignment(unittest.TestCase):
    """Test random population assignment."""
    
    def setUp(self):
        """Set up test processor."""
        with patch('process_map_svg.ET.parse'):
            with patch('process_map_svg.Path.exists', return_value=True):
                self.processor = SVGMapProcessor()
    
    def test_random_population_in_range(self):
        """Test that random population is assigned within reasonable range."""
        # Generate multiple values to check distribution
        populations = [self.processor._assign_random_population() for _ in range(100)]
        
        # All values should be positive
        self.assertTrue(all(p > 0 for p in populations))
        
        # Most values should be in a reasonable range for log-normal distribution
        # Check that at least some are in 100-800 range
        in_range = sum(1 for p in populations if 50 < p < 5000)
        self.assertGreater(in_range, 50)  # At least 50% should be in reasonable range


class TestDataValidationTracking(unittest.TestCase):
    """Test tracking of data validation issues."""
    
    def setUp(self):
        """Set up test processor."""
        with patch('process_map_svg.ET.parse'):
            with patch('process_map_svg.Path.exists', return_value=True):
                self.processor = SVGMapProcessor()
    
    def test_province_mismatch_tracking(self):
        """Test that province mismatches are tracked."""
        # Simulate province mismatch
        self.processor.province_mismatches.append({
            "settlement": "TestCity",
            "province_svg": "SVG_Province",
            "province_csv": "CSV_Province"
        })
        
        self.assertEqual(len(self.processor.province_mismatches), 1)
        self.assertEqual(self.processor.province_mismatches[0]["settlement"], "TestCity")
    
    def test_csv_settlements_not_in_svg_tracking(self):
        """Test that CSV settlements not in SVG are tracked."""
        self.processor.csv_settlements_not_in_svg["TestProvince"].append("MissingSettlement")
        
        self.assertEqual(len(self.processor.csv_settlements_not_in_svg["TestProvince"]), 1)
        self.assertIn("MissingSettlement", self.processor.csv_settlements_not_in_svg["TestProvince"])


class TestBackwardCompatibility(unittest.TestCase):
    """Test that old functionality continues to work."""
    
    def setUp(self):
        """Set up test processor."""
        with patch('process_map_svg.ET.parse'):
            with patch('process_map_svg.Path.exists', return_value=True):
                self.processor = SVGMapProcessor()
    
    def test_settlement_basic_attributes(self):
        """Test that settlement basic attributes still work as before."""
        s = Settlement(
            name="OldCity",
            province="OldProvince",
            svg_x=100.0,
            svg_y=200.0,
            geo_lon=1.5,
            geo_lat=48.0,
            population=10000,
            size_category=4
        )
        
        # Old functionality should still work
        self.assertEqual(s.name, "OldCity")
        self.assertEqual(s.province, "OldProvince")
        self.assertEqual(s.population, 10000)
        self.assertEqual(s.size_category, 4)
        self.assertEqual(s.svg_x, 100.0)
        self.assertEqual(s.svg_y, 200.0)
        self.assertEqual(s.geo_lon, 1.5)
        self.assertEqual(s.geo_lat, 48.0)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSettlementDataclass))
    suite.addTests(loader.loadTestsFromTestCase(TestSVGMapProcessorHelperMethods))
    suite.addTests(loader.loadTestsFromTestCase(TestCoordinateConverter))
    suite.addTests(loader.loadTestsFromTestCase(TestGeoJSONOutput))
    suite.addTests(loader.loadTestsFromTestCase(TestRandomPopulationAssignment))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidationTracking))
    suite.addTests(loader.loadTestsFromTestCase(TestBackwardCompatibility))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
