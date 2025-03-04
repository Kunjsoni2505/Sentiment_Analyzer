from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),  # Home Page with Text & Image Options
    path('text/', views.process_text, name='process_text'),  # Text Sentiment Analysis
    path('result/', views.result, name='result'),  # Text Sentiment Result Page
    path('process_csv/', views.process_csv, name='process_csv'),  # CSV Processing
    path('image/', views.image_analysis, name='image_analysis'),  # Image Analysis Page
    path('upload_image/', views.upload_image, name='upload_image'),  # Upload Image Processing
    path('webcam/', views.webcam_stream, name='webcam_stream'),  # Real-Time Webcam Analysis
    path("image_result/<int:image_id>/", views.image_result, name="image_result"),
    path('webcam_predict/', views.webcam_predict, name='webcam_predict'),
]
# urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)