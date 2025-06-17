#!/bin/bash

# Initialize logging (console only)
log_message() {
    local message="$1"
    local level="${2:-INFO}"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $level: $message"
}

# Set terminal title
echo -ne "\033]0;Chat-Linux-Gguf\007"

# Determine script directory and change to it
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || {
    log_message "Error: Failed to change to script directory." "ERROR"
    sleep 3
    exit 1
}
log_message "Changed to script directory: $SCRIPT_DIR"
sleep 1

# Check for required files
for file in launcher.py installer.py validater.py; do
    if [ ! -f "$file" ]; then
        log_message "Error: Required file $file not found in $SCRIPT_DIR" "ERROR"
        sleep 3
        exit 1
    fi
done
sleep 1

# Detect terminal width with fallback
TERM_WIDTH=$(tput cols 2>/dev/null || echo 80)
if [ "$TERM_WIDTH" -ge 120 ]; then
    MENU_WIDTH=120
else
    MENU_WIDTH=80
fi
log_message "Terminal width: $TERM_WIDTH, Menu width: $MENU_WIDTH"
sleep 1

# Separator functions
display_separator_thick() {
    if [ "$MENU_WIDTH" -eq 120 ]; then
        echo "======================================================================================================================="
    else
        echo "==============================================================================="
    fi
}

display_separator_thin() {
    if [ "$MENU_WIDTH" -eq 120 ]; then
        echo "-----------------------------------------------------------------------------------------------------------------------"
    else
        echo "-------------------------------------------------------------------------------"
    fi
}

# Menu functions
main_menu_80() {
    clear
    display_separator_thick
    echo "    Chat-Linux-Gguf: Bash Menu"
    display_separator_thick
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo "    1. Run Main Program"
    echo ""
    echo "    2. Run Installation"
    echo ""
    echo "    3. Run Validation"
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    display_separator_thick
    read -p "Selection; Menu Options = 1-3, Exit Bash = X: " choice
    choice=${choice//[[:space:]]/} # Trim whitespace
    process_choice
}

main_menu_120() {
    clear
    display_separator_thick
    echo "                                             _________  .____      ________                                          "
    echo "                                             \_   ___ \ |    |    /  _____/                                          "
    echo "                                       ______/    \  \/ |    |   /   \  ___  ______                                  "
    echo "                                      /_____/\     \____|    |___\    \_\  \/_____/                                  " 
    echo "                                              \______  /|_______ \\______  /                                         " 
    echo "                                                     \/         \/       \/                                          "                           
    display_separator_thin
    echo "    Chat-Linux-Gguf: Bash Menu                                       "
    display_separator_thick
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo "    1. Run Main Program"
    echo ""
    echo "    2. Run Installation"
    echo ""
    echo "    3. Run Validation"
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    display_separator_thick
    read -p "Selection; Menu Options = 1-3, Exit Bash = X: " choice
    choice=${choice//[[:space:]]/} # Trim whitespace
    process_choice
}

# Choice processing with retry limit
MAX_RETRIES=3
retry_count=0
process_choice() {
    case "$choice" in
        1)
            run_main_program
            ;;
        2)
            run_installation
            ;;
        3)
            run_validation
            ;;
        X|x)
            log_message "Closing Chat-Linux-Gguf..."
            sleep 1
            exit 0
            ;;
        *)
            ((retry_count++))
            log_message "Invalid selection. Attempt $retry_count of $MAX_RETRIES." "WARNING"
            sleep 1
            if [ "$retry_count" -ge "$MAX_RETRIES" ]; then
                log_message "Maximum retries reached. Exiting." "ERROR"
                sleep 3
                exit 1
            fi
            main_menu
            ;;
    esac
}

# Check if running interactively
pause_if_interactive() {
    if [[ -t 0 ]]; then
        read -p "Press Enter to continue..."
    fi
}

# Option handlers
run_main_program() {
    clear
    display_separator_thick
    if [ "$MENU_WIDTH" -eq 120 ]; then
        echo "                                     Chat-Linux-Gguf: Launcher                                      "
    else
        echo "    Chat-Linux-Gguf: Launcher"
    fi
    display_separator_thick
    echo ""
    echo "Checking environment..."
    # Check virtual environment and configuration
    if [ ! -f ".venv/bin/python" ] || [ ! -f "data/persistent.json" ]; then
        log_message "Error: Virtual environment or configuration file missing. Please run installation and validation first." "ERROR"
        sleep 3
        pause_if_interactive
        main_menu
        return
    fi
    # Run validation before launching
    source .venv/bin/activate
    python3 validater.py
    local validation_exit_code=$?
    deactivate
    if [ $validation_exit_code -ne 0 ]; then
        log_message "Error: Validation failed. Cannot run main program." "ERROR"
        sleep 3
        pause_if_interactive
        main_menu
        return
    fi
    echo "Starting Chat-Linux-Gguf..."
    sleep 1
    # Activate virtual environment
    source .venv/bin/activate
    log_message "Activated: .venv"
    sleep 1
    # Set PYTHONUNBUFFERED for unbuffered output
    export PYTHONUNBUFFERED=1
    # Run the launcher script
    python3 -u launcher.py
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_message "Error launching Chat-Linux-Gguf (exit code: $exit_code)" "ERROR"
        sleep 3
    fi
    # Deactivate virtual environment
    deactivate
    log_message "DeActivated: .venv"
    sleep 1
    unset PYTHONUNBUFFERED
    pause_if_interactive
    main_menu
}

run_installation() {
    clear
    display_separator_thick
    if [ "$MENU_WIDTH" -eq 120 ]; then
        echo "                                     Chat-Linux-Gguf: Installer                                      "
    else
        echo "    Chat-Linux-Gguf: Installer"
    fi
    display_separator_thick
    echo ""
    echo "Note: Installation may require sudo for system dependencies."
    echo "WARNING: This will delete ./data and ./.()['venv directories."
    if [[ -t 0 ]]; then
        read -p "Continue? (y/N): " confirm
        if [ "${confirm,,}" != "y" ]; then
            log_message "Installation cancelled by user" "INFO"
            sleep 1
            main_menu
            return
        fi
        sleep 1
    fi
    echo "Preparing installation..."
    # Remove existing data and virtual environment
    rm -rf data
    log_message "Deleted: data"
    sleep 1
    echo "Foolproofing VENV Deletion"
    deactivate 2>/dev/null || true
    rm -rf .venv
    log_message "Deleted: .venv"
    sleep 1
    echo ""
    echo "Preparation Complete."
    sleep 1
    echo "Running Installer..."
    sleep 1
    clear
    # Run the installer script
    python3 installer.py
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_message "Error during installation (exit code: $exit_code)" "ERROR"
        sleep 3
    else
        log_message "Installation completed" "INFO"
        sleep 1
    fi
    # Ensure venv is deactivated
    deactivate 2>/dev/null || true
    log_message "DeActivated: .venv (if active)"
    sleep 1
    unset PYTHONUNBUFFERED
    pause_if_interactive
    main_menu
}

run_validation() {
    clear
    display_separator_thick
    if [ "$MENU_WIDTH" -eq 120 ]; then
        echo "                                     Chat-Linux-Gguf: Library Validation                                      "
    else
        echo "    Chat-Linux-Gguf: Library Validation"
    fi
    display_separator_thick
    echo ""
    echo "Running Library Validation..."
    sleep 1
    # Check virtual environment
    if [ ! -f ".venv/bin/python" ]; then
        log_message "Error: Virtual environment not found or invalid at .venv" "ERROR"
        sleep 3
        pause_if_interactive
        main_menu
        return
    fi
    # Activate virtual environment
    source .venv/bin/activate
    log_message "Activated: .venv"
    sleep 1
    # Run the validation script
    python3 validater.py
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_message "Error during validation (exit code: $exit_code)" "ERROR"
        sleep 3
    else
        log_message "Validation completed" "INFO"
        sleep 1
    fi
    # Deactivate virtual environment
    deactivate
    log_message "DeActivated: .venv"
    sleep 1
    pause_if_interactive
    main_menu
}

# Main menu router
main_menu() {
    if [ "$MENU_WIDTH" -eq 120 ]; then
        main_menu_120
    else
        main_menu_80
    fi
}

# Start the script
log_message "Starting Chat-Linux-Gguf Bash script"
sleep 1
main_menu