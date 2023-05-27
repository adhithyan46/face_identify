from django.urls import path

from . import views, user_views

urlpatterns = [
    path('', views.index, name='index'),
    path('home/',views.home,name='home'),
    path('Content_add/',user_views.Content_add,name='Content_add'),
    path('Contentt/', user_views.Contentt, name='Contentt'),
    path('Content_admin',views.Content_admin,name='Content_admin'),
    path('reply_Content/<int:id>/', views.reply_Content, name='reply_Content'),
    path('video_stream/', views.video_stream, name='video_stream'),
    path('add_photos/', views.add_photos, name='add_photos'),
    path('click_photos/<int:emp_id>/', views.click_photos, name='click_photos'),
    path('train_model/', views.train_model, name='train_model'),
    path('detected/', views.detected, name='detected'),
    path('detected_out/', views.detected_out, name='detected_out'),
    path('identify/', views.identify, name='identify'),
    path('add_emp/', views.add_emp, name='add_emp'),
    path('attendece_rep/', views.attendece_rep, name='attendece_rep'),
    path('attendece_rep2/', views.attendece_rep2, name='attendece_rep2'),
    path('employee_view/', views.employee_view, name='employee_view'),
    path('employee_update/<int:id>/', views.employee_update, name='employee_update'),
    path('employee_delete/<int:id>/', views.employee_delete, name='employee_delete'),
    path('report/',views.reportt,name='report'),
    path('person/',views.person,name='person'),
    path('logout_view/',views.logout_view,name='logout_view'),
    path('login_view/',views.login_view,name='login_view'),
    path('user_profile/', user_views.user_profile, name='user_profile'),
    path('personal_report/', user_views.personal_report, name='personal_report'),
    path('generate_pdf3/', views.generate_pdf3, name='generate_pdf3'),
    path('generate_pdf4/', user_views.generate_pdf4, name='generate_pdf4'),
    path('attendance_pdf/', views.attendance_pdf, name='attendance_pdf'),
    path('generate_pdf/', views.generate_pdf, name='generate_pdf'),
    path('report_pdf/', views.report_pdf, name='report_pdf'),
    path('attendance_list/', views.attendance_list, name='attendance_list'),
    path('upload_photos/<int:id>/', views.upload_photos, name='upload_photos'),
    path('generate_unique_filename/', views.generate_unique_filename, name='generate_unique_filename')
]