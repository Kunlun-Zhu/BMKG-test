#!/usr/bin/env python
import os
import re
import sys
import json

import pycodestyle
from github import Github


class MyReport(pycodestyle.BaseReport):
    """Collect and print the results of the checks."""

    def __init__(self, options):
        super(MyReport, self).__init__(options)
        self._fmt = pycodestyle.REPORT_FORMAT.get(options.format.lower(),
                                                  options.format)
        self._repeat = options.repeat
        self._show_source = options.show_source
        self._show_pep8 = options.show_pep8
        self.reports = []

    def init_file(self, filename, lines, expected, line_offset):
        """Signal a new file."""
        return super(MyReport, self).init_file(
            filename, lines, expected, line_offset)

    def error(self, line_number, offset, text, check):
        """Report an error, according to options."""
        code = super(MyReport, self).error(line_number, offset,
                                           text, check)
        if code and (self.counters[code] == 1 or self._repeat):
            self.reports.append(
                (self.filename, line_number, offset, code, text[5:], check.__doc__))
        return code

    def get_file_results(self):
        return self.file_errors


if __name__ == "__main__":
    style = pycodestyle.StyleGuide(max_line_length=120, quiet=True, reporter=MyReport)
    result = style.check_files(["bmkg"])
    if result.total_errors != 0:
        print("Found errors!")
        g = Github(os.environ['GITHUB_TOKEN'])
        with open(os.environ['GITHUB_EVENT_PATH']) as f:
            data = json.load(f)
        try:
            if data['pull_request']['comments_url']:
                # we are in PR
                # try to comment
                repo = g.get_repo(data['repository']['full_name'])
                pr = repo.get_pull(data['pull_request']['number'])
                comments = [
                    {
                        "path": i[0],
                        "body": i[4],
                        "line": i[1]
                    }
                    for i in result.reports
                ]
                pr.create_review(comments=comments, event="REQUEST_CHANGES")
        except KeyError:
            # not in PR
            pass
        except Exception as e:
            print(e)
        exit(1)
    else:
        exit(0)
