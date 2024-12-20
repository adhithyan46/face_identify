# Generated by Django 5.1.3 on 2024-12-17 07:15

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0012_rep_name_report_name_alter_detected_in_name_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='detected_out',
            name='photo',
            field=models.ImageField(default='app/facerec/detected_out/noimg.png', upload_to='detected_out/'),
        ),
        migrations.AlterField(
            model_name='rep',
            name='emp_id',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='reports', to='app.employee'),
        ),
    ]
