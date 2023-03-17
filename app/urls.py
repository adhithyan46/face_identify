from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_stream/', views.video_stream, name='video_stream'),
    path('add_photos/', views.add_photos, name='add_photos'),
    path('add_photos/<slug:emp_id>/', views.click_photos, name='click_photos'),
    path('train_model/', views.train_model, name='train_model'),
    path('detected/', views.detected, name='detected'),
    path('detected_out/', views.detected_out, name='detected_out'),
    path('identify/', views.identify, name='identify'),
    path('add_emp/', views.add_emp, name='add_emp'),
    path('attendece_rep/', views.attendece_rep, name='attendece_rep'),
    path('logout/',views.logout,name='logout')
]