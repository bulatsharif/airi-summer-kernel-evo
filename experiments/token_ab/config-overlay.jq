.[0] * .[1]
| .instructions = [$common, $specific]
| .permission.read["work/**"] = "deny"
| .permission.read[$workspace_pattern] = "allow"
| .permission.edit["work/*/submission.py"] = "deny"
| .permission.edit[$candidate] = "allow"
| .permission.bash["python -m cute_harness *"] = "deny"
| .permission.bash["python.exe -m cute_harness *"] = "deny"
| .permission.bash["py -m cute_harness *"] = "deny"
| .permission.bash["python -m cute_harness check *"] = "allow"
| .permission.bash["python.exe -m cute_harness check *"] = "allow"
| .permission.bash["py -m cute_harness check *"] = "allow"
| if $arm == "local"
  then .skills = {"paths": [$skill_dir]}
  else del(.skills)
  end
