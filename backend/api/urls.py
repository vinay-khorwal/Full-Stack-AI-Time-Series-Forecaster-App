# backend/api/urls.py
from django.urls import path
from .views import AnalyzeFileView, PredictForecastView

urlpatterns = [
    # Endpoint for the initial file upload and column analysis
    path('analyze/', AnalyzeFileView.as_view(), name='analyze_file'),
    # Endpoint to run the forecast after the user selects a column
    path('predict/', PredictForecastView.as_view(), name='predict_forecast'),
]