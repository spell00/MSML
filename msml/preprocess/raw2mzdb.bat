@ECHO OFF
set spd=%1%
set group=%2%
IF [%1%]==[] (set spd=200)
IF [%2%]==[] (set group=test)
FOR /R "..\..\..\..\resources\raw\%spd%spd\%group%" %%G IN (*.raw) DO (
    START .\raw2mzDB_0.9.10_build20170802\raw2mzDB.exe -i "%%G" -o "..\..\..\..\resources\raw\%spd%spd\%group%\%%~nG.mzDB" -f 1-2 -a "dia")
GOTO TEST

:TEST
tasklist.exe | findstr "raw2mzDB.exe" > nul
cls
if errorlevel 1 ( GOTO NEXT ) else ( CALL timeout 10 /nobreak > nul && GOTO TEST )

:NEXT
if not exist "..\..\..\..\resources\mzdb\%spd%spd\%group%" MD ..\..\..\..\resources\mzdb\%spd%spd\%group%
FOR /R ..\..\..\..\resources\raw\%spd%spd\%group% %%G IN (*.mzdb) DO (
    MOVE "..\..\..\..\resources\raw\%spd%spd\%group%\%%~nG.mzDB" "..\..\..\..\resources\mzdb\%spd%spd\%group%\%%~nG.mzDB")
