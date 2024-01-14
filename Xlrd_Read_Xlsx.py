import os
import xlrd
os.chdir('C:/ProgramData/Anaconda3/Scripts/IPBA_Project')
datalist=[]
i=0
try:
    wb = xlrd.open_workbook('DataSet.xls')
    sheet = wb.sheet_by_name('Keyword_Value')
    while (sheet.cell(i, 0).value) != None:
        datalist.append(sheet.cell(i, 0).value)
        # memotable.append(sheet.cell(i,1).value)
        i += 1
    datalist.pop(0)
except FileNotFoundError:
    pass
except IndexError:
    pass

print(datalist)