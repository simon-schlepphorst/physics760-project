#rwildcard=$(wildcard $1$2) $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2))
rwildcard=$(wildcard $(addsuffix $2, $1)) $(foreach d,$(wildcard $(addsuffix *, $1)),$(call rwildcard,$d/,$2))

all_configs := $(call rwildcard,runs/,config.ini)
all_simulations :=$(all_configs:config.ini=simulation.npz)

all: $(all_simulations)

%simulation.npz: %config.ini
	python3 ising.py $<
