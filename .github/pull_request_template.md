<!-- If this PR closes a JIRA ticket, make sure the title starts with the JIRA issue number,
for example JP-1234: <Fix a bug> -->

Resolves [JP-nnnn](https://jira.stsci.edu/browse/JP-nnnn)
Resolves [RCAL-nnnn](https://jira.stsci.edu/browse/RCAL-nnnn)

<!-- If this PR closes a GitHub issue, reference it here by its number -->

Closes #

<!-- describe the changes comprising this PR here -->

This PR addresses ...

<!-- if you can't perform these tasks due to permissions, please ask a maintainer to do them -->
## Tasks
- [ ] update or add relevant tests
- [ ] update relevant docstrings and / or `docs/` page
- [ ] Does this PR change any API used downstream? (if not, label with `no-changelog-entry-needed`)
  - [ ] write news fragment(s) in `changes/`: `echo "changed something" > changes/<PR#>.<changetype>.rst` (see below for change types)
  - [ ] run regression tests with this branch installed (`"git+https://github.com/<fork>/stcal@<branch>"`)
    - [ ] [`jwst` regression test](https://github.com/spacetelescope/RegressionTests/actions/workflows/jwst.yml) 
    - [ ] [`romancal` regression test](https://github.com/spacetelescope/RegressionTests/actions/workflows/romancal.yml)

<details><summary>news fragment change types...</summary>

- ``changes/<PR#>.apichange.rst``: change to public API
- ``changes/<PR#>.bugfix.rst``: fixes an issue
- ``changes/<PR#>.general.rst``: infrastructure or miscellaneous change
</details>
