#!/bin/sh
# Gives the twin server (i-05869be7e81e57ac0) permission to call Claude on Bedrock, and nothing else.
set -e
cd "$(dirname "$0")"
aws iam create-role --role-name twin-bedrock --assume-role-policy-document file://bedrock-trust.json --query Role.Arn --output text
aws iam put-role-policy --role-name twin-bedrock --policy-name bedrock-claude-only --policy-document file://bedrock-policy.json
aws iam create-instance-profile --instance-profile-name twin-bedrock --query InstanceProfile.Arn --output text
aws iam add-role-to-instance-profile --instance-profile-name twin-bedrock --role-name twin-bedrock
sleep 10
aws ec2 associate-iam-instance-profile --region us-east-1 --instance-id i-05869be7e81e57ac0 \
  --iam-instance-profile Name=twin-bedrock --query IamInstanceProfileAssociation.State --output text
