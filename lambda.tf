
# ------------------------------------------------------------------------------
# IAM Role for Lambda
# ------------------------------------------------------------------------------

resource "aws_iam_role" "lambda_exec" {
  name = "mtg_price_checker_lambda_role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "lambda_exec_policy" {
  name = "mtg_price_checker_lambda_policy"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "dynamodb:PutItem",
          "dynamodb:GetItem",
          "dynamodb:Query",
          "dynamodb:Scan",
          "dynamodb:BatchWriteItem"
        ]
        Effect   = "Allow"
        Resource = aws_dynamodb_table.mtg_prices.arn
      },
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Effect   = "Allow"
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# ------------------------------------------------------------------------------
# Build / Package Lambda
# ------------------------------------------------------------------------------

resource "null_resource" "install_dependencies" {
  triggers = {
    requirements_hash = filemd5("${path.module}/requirements.txt")
    script_hash       = filemd5("${path.module}/mtg_price_checker.py")
  }

  provisioner "local-exec" {
    command = <<EOT
      mkdir -p ${path.module}/lambda_build
      pip install -r ${path.module}/requirements.txt -t ${path.module}/lambda_build --upgrade
      cp ${path.module}/mtg_price_checker.py ${path.module}/lambda_build/
      cp ${path.module}/cards.json ${path.module}/lambda_build/
    EOT
  }
}

data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = "${path.module}/lambda_build"
  output_path = "${path.module}/mtg_price_checker.zip"

  depends_on = [null_resource.install_dependencies]
}

# ------------------------------------------------------------------------------
# Lambda Function
# ------------------------------------------------------------------------------

resource "aws_lambda_function" "mtg_price_checker" {
  function_name    = "mtg_price_checker"
  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  role             = aws_iam_role.lambda_exec.arn
  handler          = "mtg_price_checker.lambda_handler"
  runtime          = "python3.9" # Or your preferred version
  timeout          = 60
  memory_size      = 128

  depends_on = [
    aws_iam_role_policy.lambda_exec_policy,
    data.archive_file.lambda_zip
  ]
}

# ------------------------------------------------------------------------------
# EventBridge (CloudWatch Events) Scheduled Trigger
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_event_rule" "daily_trigger" {
  name                = "mtg_price_checker_daily"
  description         = "Triggers MTG Price Checker once a day"
  schedule_expression = "rate(1 day)"
}

resource "aws_cloudwatch_event_target" "check_prices_target" {
  rule      = aws_cloudwatch_event_rule.daily_trigger.name
  target_id = "lambda"
  arn       = aws_lambda_function.mtg_price_checker.arn
}

resource "aws_lambda_permission" "allow_cloudwatch" {
  statement_id  = "AllowExecutionFromCloudWatch"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.mtg_price_checker.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.daily_trigger.arn
}
