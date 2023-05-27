import datetime

from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.template.loader import get_template
from django.contrib import messages
from xhtml2pdf import pisa

from app.forms import ContentForm
from app.models import Content, Employee, Detected_in, Detected_out

@login_required(login_url='/app/login/')
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

@login_required(login_url='/app/login/')
def Contentt(request):
    n = Content.objects.filter(user=request.user)
    return render(request, 'app/content.html', {'Content': n})


@login_required(login_url='/app/login/')
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
        return render(request, 'admintemp/personal_report.html', context)
@login_required(login_url='/app/login/')
def generate_pdf4(request):
    # get the template
    template = get_template('admintemp/personal_report.html')

    # get the context data
    user = request.user
    employee = Employee.objects.get(user=user)
    date_formatted = datetime.datetime.today().date()
    date = request.GET.get('search_box', None)
    if date is not None:
        date_formatted = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    report_in = Detected_in.objects.filter(emp_id=employee, entry__date=date_formatted).order_by('emp_id_id').reverse()
    report_out = Detected_out.objects.filter(emp_id=employee, out__date=date_formatted).order_by('emp_id_id').reverse()

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

    context = {
        'report_data': report_data,
        # 'date': date_formatted,
    }

    # render the template with the context data
    html = template.render(context)

    # create a PDF object
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'filename="personal_report.pdf"'

    # create the PDF
    pisa_status = pisa.CreatePDF(
        html, dest=response)

    # return the PDF object
    if pisa_status.err:
        return HttpResponse('Failed to generate PDF')
    return response
