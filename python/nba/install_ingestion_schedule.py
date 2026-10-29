"""Preview macOS launch agents; --install enables daily stats and optional injuries."""

import argparse
import os
from pathlib import Path
import plistlib
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def configurations(root=ROOT, injuries=False, profiles='weekly'):
    root=Path(root).absolute()
    logs=root/'python/data/raw/ingestion/logs'
    python=root/'.venv/bin/python'
    jobs=[('com.nba-betting.daily-ingestion',
           [str(python),'-B','-u',str(root/'python/nba/ingest_daily.py'),'--apply','--profiles',profiles],
           dict(StartCalendarInterval=dict(Hour=7,Minute=15)))]
    if injuries:
        jobs.append(('com.nba-betting.injury-reports',
                     [str(python),'-B','-u',str(root/'python/nba/collect_injuries.py'),'--apply'],
                     dict(StartCalendarInterval=[dict(Hour=h,Minute=m) for h in range(10,24) for m in (0,15,30,45)])))
    return [dict(Label=label,ProgramArguments=args,WorkingDirectory=str(root),
                 StandardOutPath=str(logs/f'{label}.out.log'),StandardErrorPath=str(logs/f'{label}.err.log'),
                 RunAtLoad=False,ThrottleInterval=60,**schedule) for label,args,schedule in jobs]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install',action='store_true')
    parser.add_argument('--injuries',action='store_true',help='Also poll reports every 15 minutes, 10am–11:45pm local time')
    parser.add_argument('--profiles',choices=['none','weekly','daily'],default='weekly')
    args=parser.parse_args()
    if sys.platform!='darwin':
        parser.error('This installer is for macOS launchd; use the documented Python commands with your scheduler elsewhere')
    jobs=configurations(injuries=args.injuries,profiles=args.profiles)
    destination=Path.home()/'Library/LaunchAgents'
    for job in jobs:
        data=plistlib.dumps(job)
        print(data.decode(),flush=True)
    if not args.install:
        print('Preview only. Add --install to register these jobs. Schedules use the Mac’s local timezone.')
        return
    # Validate every destination before changing any of them.
    for job in jobs:
        path=destination/f'{job["Label"]}.plist'
        if path.exists() and plistlib.loads(path.read_bytes()).get('WorkingDirectory')!=str(ROOT):
            raise ValueError(f'Refusing to replace a job belonging to another directory: {path}')
        if not Path(job['ProgramArguments'][0]).is_file():
            raise ValueError('Project virtual environment is missing')
    destination.mkdir(parents=True,exist_ok=True)
    (ROOT/'python/data/raw/ingestion/logs').mkdir(parents=True,exist_ok=True)
    domain=f'gui/{os.getuid()}'
    for job in jobs:
        path=destination/f'{job["Label"]}.plist'
        if path.exists():
            path.with_suffix('.plist.previous').write_bytes(path.read_bytes())
        # bootout can fail when a job is not registered yet.
        subprocess.run(['launchctl','bootout',f'{domain}/{job["Label"]}'],capture_output=True)
        path.write_bytes(plistlib.dumps(job))
        path.chmod(0o600)
        subprocess.run(['launchctl','bootstrap',domain,str(path)],check=True)
        subprocess.run(['launchctl','enable',f'{domain}/{job["Label"]}'],check=True)
        subprocess.run(['launchctl','print',f'{domain}/{job["Label"]}'],check=True,stdout=subprocess.DEVNULL)
        print(f'Installed and verified: {path}',flush=True)


if __name__=='__main__':
    main()
