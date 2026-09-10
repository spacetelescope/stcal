<!-- If this PR addresses a JIRA ticket: -->
<!-- Resolves [JP-nnnn](https://jira.stsci.edu/browse/JP-nnnn) -->
<!-- Resolves [RCAL-nnnn](https://jira.stsci.edu/browse/RCAL-nnnn) -->

<!-- If this PR will close an existing GitHub issue (that is not already attached to a JIRA ticket): -->
<!-- Closes # -->

<!-- Describe your changes here: -->

## Description

This change ...

<!-- If you can't perform these tasks due to permissions, reach out to a maintainer. -->

## Tasks

- [ ] update or add relevant tests
- [ ] update relevant docstrings and / or `docs/` page
- [ ] Does this PR change any API used downstream? (if not, label with `no-changelog-entry-needed`)
  - [ ] write news fragment(s) in `changes/`: `echo "changed something" > changes/<PR#>.<changetype>.rst` (see [changelog readme](https://github.com/spacetelescope/stcal/blob/main/changes/README.rst) for instructions)
  - [ ] if your change breaks existing functionality, also add a `changes/<PR#>.breaking.rst` news fragment
- [ ] run regression tests with this branch installed (`"git+https://github.com/<fork>/stcal@<branch>"`)
  - [ ] [`jwst` regression test](https://github.com/spacetelescope/RegressionTests/actions/workflows/jwst.yml)
  - [ ] [`romancal` regression test](https://github.com/spacetelescope/RegressionTests/actions/workflows/romancal.yml)

## Generative AI Usage Disclosure

<!-- If generative AI or LLMs were used in the process of making this change, describe their use here. -->
<!-- Otherwise, indicate "No genAI tools used". -->
