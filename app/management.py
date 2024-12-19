from .models import Attendance
from django.db.models import Count

# Identify duplicate entries
duplicates = (
    Attendance.objects.values('employee', 'date')
    .annotate(count=Count('id'))
    .filter(count__gt=1)
)

for duplicate in duplicates:
    # Filter duplicate records
    records = Attendance.objects.filter(employee=duplicate['employee'], date=duplicate['date'])
    # Keep the first record and delete the others
    records.exclude(id=records.first().id).delete()

print("Duplicates removed.")

