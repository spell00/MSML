@ECHO OFF
set spd=%1%
set experiment=%2%
IF [%1%]==[] (set spd=200)
IF [%2%]==[] (set experiment=20220706_Data_ML02\Data_FS)
FOR /R "..\..\..\..\resources\%experiment%\raw\%spd%spd\" %%G IN (*.raw) DO (
    START .\raw2mzDB_0.9.10_build20170802\raw2mzDB.exe -i "%%G" -o "..\..\..\..\resources\%experiment%\raw\%spd%spd\%%~nG.mzDB" -f 1-2 -a "dia")
GOTO TEST

:TEST
tasklist.exe | findstr "raw2mzDB.exe" > nul
cls
if errorlevel 1 ( GOTO NEXT ) else ( CALL timeout 10 /nobreak > nul && GOTO TEST )

:NEXT
if not exist "..\..\..\..\resources\%experiment%\mzdb\%spd%spd\" MD ..\..\..\..\resources\%experiment%\mzdb\%spd%spd
FOR /R ..\..\..\..\resources\%experiment%\raw\%spd%spd %%G IN (*.mzdb) DO (
    MOVE "..\..\..\..\resources\%experiment%\raw\%spd%spd\%%~nG.mzDB" "..\..\..\..\resources\%experiment%\mzdb\%spd%spd\%%~nG.mzDB")
