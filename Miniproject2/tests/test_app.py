import pytest
from agatsas_miniproject2.app import find_points_of_interest_tool,get_weather_forecast_tool


def test_weather_tool():
    """checks for key words in weather forecast based on input"""
    output = get_weather_forecast_tool.invoke({"city": "Berlin", "days": 2})
    assert "Berlin" in output
    assert "Forecast for" in output
    assert "°C" in output

def test_points_of_interest_tool():
    """checks if key words are in point of interest based on specific input"""
    output = find_points_of_interest_tool.invoke({"city": "Athens", "category": "museums"})
    assert "Athens" in output
    assert "museums" in output
    assert "indoor" or "outdoor" in output
    


def test_weather_tool_for_request_error():
    """ returns an error message if it fails to retrieve the data"""
    output = get_weather_forecast_tool.invoke({"city": " ", "days": 2})
    assert "Failed to fetch weather data" in output or "An error occurred" in output

