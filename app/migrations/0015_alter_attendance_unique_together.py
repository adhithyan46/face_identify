# Generated by Django 5.1.3 on 2024-12-19 03:51

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0014_alter_attendance_unique_together'),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name='attendance',
            unique_together=set(),
        ),
    ]
