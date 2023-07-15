import datetime

from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.template.loader import get_template, render_to_string
from django.contrib import messages
from xhtml2pdf import pisa
from app.forms import ContentForm
from app.models import Content, Employee, Detected_in, Detected_out
from io import BytesIO
# @login_required(login_url='/app/login/')
def Content_add(request):
    form = ContentForm()
    u = request.user
    if request.method == 'POST':
        form = ContentForm(request.POST)
        if form.is_valid():
            obj = form.save(commit=False)
            obj.user = u
            obj.save()
            messages.info(request, 'Content Registered Successfully')
            return redirect('Contentt')
    else:
        form = ContentForm()
    return render(request, 'app/content_add.html', {'form': form})

# @login_required(login_url='/app/login/')
def Contentt(request):
    n = Content.objects.filter(user=request.user)
    return render(request, 'app/content.html', {'Content': n})


# @login_required(login_url='/app/login/')
def user_profile(request):
    u = request.user
    profile = Employee.objects.filter(user=u)
    return render(request, 'app/user_profile.html', {'profile': profile})

@login_required(login_url='/app/login/')
def personal_report(request):
    if request.method == 'GET':
        user = request.user
        employee = Employee.objects.get(user=user)
        date_formatted = datetime.datetime.today().date()
        date = request.GET.get('search_box', None)
        if date is not None:
            date_formatted = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        report_in = Detected_in.objects.filter(emp_id=employee, entry__date=date_formatted).order_by(
            'emp_id_id').reverse()
        report_out = Detected_out.objects.filter(emp_id=employee, out__date=date_formatted).order_by(
            'emp_id_id').reverse()

        report_data = []
        for rep1, rep2 in zip(report_in, report_out):
            if rep1.emp_id_id == rep2.emp_id_id:
                total_time = rep2.out - rep1.entry
                total_time_str = str(datetime.timedelta(seconds=total_time.seconds))
                report_data.append({
                    # 'employee_id': user.id,
                    # 'employee_name': user.name,
                    'entry_time': rep1.entry,
                    'exit_time': rep2.out,
                    'total_time': total_time_str,
                })
        print(user)
        context = {
            'report_data': report_data,
            # 'date': date_formatted,
        }

        if 'generate_pdf' in request.GET:
            # Generate PDF
            html_string = render_to_string('admintemp/personal_report.html', context)
            result=BytesIO()
            pdf = pisa.pisaDocument(BytesIO(html_string.encode("UTF-8")), result)

            # Create HTTP response with PDF attachment
            response = HttpResponse(content_type='application/pdf')
            response['Content-Disposition'] = 'attachment; filename="personal_report.pdf"'
            response.write(result.getvalue())

            return response

        return render(request, 'admintemp/personal_report.html', context)


from django.db.models import Min, Max
from django.http import HttpResponse
from django.shortcuts import render
from django.template.loader import get_template
from io import BytesIO
import datetime
import django.template.loader as loader
from xhtml2pdf import pisa

from django.db.models import Min, Max

from django.db.models import Min, Max

from django.db.models import Min, Max

from django.contrib.auth.decorators import login_required


@login_required(login_url='/app/login/')
def personal_report2(request):
    if request.method == 'GET':
        user = request.user
        employee = Employee.objects.get(user=user)
        start_date = request.GET.get('start_date', None)
        end_date = request.GET.get('end_date', None)
        if start_date and end_date:
            start_date_formatted = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
            end_date_formatted = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

            attendance_data = []

            det_list_in = Detected_in.objects.filter(emp_id=employee,
                                                     entry__date__range=(start_date_formatted, end_date_formatted))
            det_list_out = Detected_out.objects.filter(emp_id=employee,
                                                       out__date__range=(start_date_formatted, end_date_formatted))

            date_range = set(det_list_in.values_list('entry__date', flat=True)) | set(
                det_list_out.values_list('out__date', flat=True))

            # Process each date for the employee
            for date in sorted(date_range):
                det_in = det_list_in.filter(entry__date=date).order_by('entry').first()
                det_out = det_list_out.filter(out__date=date).order_by('out').last()

                if det_in and det_out:
                    total_time = det_out.out - det_in.entry
                    total_time_str = str(datetime.timedelta(seconds=total_time.seconds))
                    attendance_data.append({
                        'employee_id': employee.id,
                        'employee_name': employee.name,
                        'date': date,
                        'entry_time': det_in.entry,
                        'exit_time': det_out.out,
                        'total_time': total_time_str,
                    })
                elif det_in:
                    attendance_data.append({
                        'employee_id': employee.id,
                        'employee_name': employee.name,
                        'date': date,
                        'entry_time': det_in.entry,
                        'exit_time': None,
                        'total_time': None,
                    })
                elif det_out:
                    attendance_data.append({
                        'employee_id': employee.id,
                        'employee_name': employee.name,
                        'date': date,
                        'entry_time': None,
                        'exit_time': det_out.out,
                        'total_time': None,
                    })

            # Sort attendance data by date
            attendance_data.sort(key=lambda x: x['date'])

            context = {
                'attendance_data': attendance_data,
                'start_date': start_date_formatted,
                'end_date': end_date_formatted,
            }
        else:
            # Dates are not selected or empty, no change occurs
            context = {}

        if 'generate_pdf' in request.GET:
            # Generate PDF
            template = get_template('app/personal_report.html')
            html = template.render(context)
            result = BytesIO()

            # Convert HTML to PDF
            pisaStatus = pisa.CreatePDF(html, dest=result)
            if pisaStatus.err:
                return HttpResponse('Error generating PDF')

            # Create HTTP response with PDF attachment
            response = HttpResponse(result.getvalue(), content_type='application/pdf')
            response['Content-Disposition'] = 'attachment; filename="personal_report.pdf"'

            return response

    else:
        # Handle GET request with initial date range selection form
        context = {}

    return render(request, 'app/personal_report2.html', context)
# @login_required(login_url='/app/login/')
# def generate_pdf4(request):
#     # get the template
#     template = get_template('admintemp/personal_report.html')
#
#     # get the context data
#     user = request.user
#     employee = Employee.objects.get(user=user)
#     date_formatted = datetime.datetime.today().date()
#     date = request.GET.get('search_box', None)
#     if date is not None:
#         date_formatted = datetime.datetime.strptime(date, "%Y-%m-%d").date()
#     report_in = Detected_in.objects.filter(emp_id=employee, entry__date=date_formatted).order_by('emp_id_id').reverse()
#     report_out = Detected_out.objects.filter(emp_id=employee, out__date=date_formatted).order_by('emp_id_id').reverse()
#
#     report_data = []
#     for rep1, rep2 in zip(report_in, report_out):
#         if rep1.emp_id_id == rep2.emp_id_id:
#             total_time = rep2.out - rep1.entry
#             total_time_str = str(datetime.timedelta(seconds=total_time.seconds))
#             report_data.append({
#                 # 'employee_id': user.id,
#                 # 'employee_name': user.name,
#                 'entry_time': rep1.entry,
#                 'exit_time': rep2.out,
#                 'total_time': total_time_str,
#             })
#
#     context = {
#         'report_data': report_data,
#         # 'date': date_formatted,
#     }
#
#     # render the template with the context data
#     html = template.render(context)
#
#     # create a PDF object
#     response = HttpResponse(content_type='application/pdf')
#     response['Content-Disposition'] = 'filename="personal_report.pdf"'
#
#     # create the PDF
#     pisa_status = pisa.CreatePDF(
#         html, dest=response)
#
#     # return the PDF object
#     if pisa_status.err:
#         return HttpResponse('Failed to generate PDF')
#     return response
