# backend/api/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import FileSystemStorage
import os
from .ml_pipeline import analyze_file_columns, run_forecasting_pipeline

class AnalyzeFileView(APIView):
    def post(self, request, *args, **kwargs):
        file_obj = request.FILES.get('file')
        if not file_obj:
            return Response({"message": "No file was uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        # Save the file to a temporary location
        fs = FileSystemStorage(location='tmp/')
        filename = fs.save(file_obj.name, file_obj)
        file_path = fs.path(filename)
        
        # Call the new analysis function from our pipeline
        analysis_results = analyze_file_columns(file_path)
        
        if analysis_results['status'] == 'error':
            os.remove(file_path) # Clean up if analysis fails
            return Response(analysis_results, status=status.HTTP_400_BAD_REQUEST)
        
        # Add the temporary filename to the response so the frontend can send it back
        analysis_results['temp_filename'] = filename
        return Response(analysis_results, status=status.HTTP_200_OK)

class PredictForecastView(APIView):
    def post(self, request, *args, **kwargs):
        filename = request.data.get('filename')
        target_column = request.data.get('target_column')

        if not filename or not target_column:
            return Response({"message": "Filename and target column are required."}, status=status.HTTP_400_BAD_REQUEST)

        file_path = os.path.join('tmp', filename)
        
        if not os.path.exists(file_path):
            return Response({"message": "The uploaded file has expired or could not be found. Please upload again."}, status=status.HTTP_404_NOT_FOUND)

        # Run the new forecasting pipeline
        forecast_results = run_forecasting_pipeline(file_path, target_column)
        
        # Clean up the file after forecasting is complete
        os.remove(file_path)
        
        if forecast_results['status'] == 'error':
            return Response(forecast_results, status=status.HTTP_400_BAD_REQUEST)
            
        return Response(forecast_results, status=status.HTTP_200_OK)