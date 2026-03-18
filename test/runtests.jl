using Test
using EFDMTempering
using CSV
using DataFrames
using JSON

@testset "EFDMTempering" begin

    @testset "Stan model paths" begin
        for sym in (:multinomial, :dm, :fdm, :efdm)
            p = stan_model_path(sym)
            @test isfile(p)
            @test endswith(p, ".stan")
        end

        @test_throws ArgumentError stan_model_path(:nonexistent)
    end

    @testset "BUNDLED_MODELS registry" begin
        @test haskey(BUNDLED_MODELS, :multinomial)
        @test haskey(BUNDLED_MODELS, :dm)
        @test haskey(BUNDLED_MODELS, :fdm)
        @test haskey(BUNDLED_MODELS, :efdm)
    end

    @testset "prepare_efdm_data – returns valid JSON" begin
        csv_path = joinpath(@__DIR__, "..", "data", "example_counts.csv")
        @test isfile(csv_path)

        json_str = EFDMTempering.prepare_efdm_data(csv_path)

        # Must be a parseable JSON string
        @test json_str isa String
        d = JSON.parse(json_str)

        @test d["N"] > 0
        @test d["D"] == 4
        @test d["K"] >= 1
        @test length(d["Y"]) == d["N"]
        @test length(d["X"]) == d["N"]
        @test length(d["n"]) == d["N"]
        @test length(d["w_hyper"]) == d["D"]
        @test d["sd_prior"] == 50.0

        # All trial counts should be positive
        @test all(d["n"] .> 0)

        # Row sums of Y should equal n
        for i in 1:d["N"]
            @test sum(d["Y"][i]) == d["n"][i]
        end
    end

    @testset "prepare_efdm_data – explicit columns" begin
        csv_path = joinpath(@__DIR__, "..", "data", "example_counts.csv")
        json_str = EFDMTempering.prepare_efdm_data(
            csv_path;
            response_cols  = ["Y1", "Y2", "Y3", "Y4"],
            covariate_cols = ["x"],
            sd_prior       = 10.0,
            w_hyper        = [2.0, 2.0, 2.0, 2.0],
        )
        d = JSON.parse(json_str)
        @test d["sd_prior"] == 10.0
        @test d["w_hyper"] == [2.0, 2.0, 2.0, 2.0]
        @test d["K"] == 2   # intercept + x
    end

    @testset "prepare_efdm_data – w_hyper length check" begin
        csv_path = joinpath(@__DIR__, "..", "data", "example_counts.csv")
        @test_throws ArgumentError EFDMTempering.prepare_efdm_data(
            csv_path; w_hyper = [1.0, 1.0]   # wrong length (D=4)
        )
    end

    @testset "_prepare_multinomial_data – returns valid JSON" begin
        csv_path = joinpath(@__DIR__, "..", "data", "example_counts.csv")
        json_str = EFDMTempering._prepare_multinomial_data(csv_path)
        @test json_str isa String
        d = JSON.parse(json_str)
        @test d["N"] > 0
        @test d["D"] == 4
        @test !haskey(d, "n")        # Multinomial model has no n vector
        @test !haskey(d, "w_hyper")  # Multinomial model has no w_hyper
    end

    @testset "dict_to_json round-trip" begin
        data = Dict{String,Any}("x" => 1, "y" => [1.0, 2.0], "z" => [[1,2],[3,4]])
        json_str = EFDMTempering.dict_to_json(data)
        @test json_str isa String
        d2 = JSON.parse(json_str)
        @test d2["x"] == 1
        @test d2["y"] == [1.0, 2.0]
    end

    @testset "load_csv – file not found" begin
        @test_throws ArgumentError EFDMTempering.load_csv("/nonexistent/file.csv")
    end

    @testset "fit_stan_with_pigeons – file checks" begin
        csv_path = joinpath(@__DIR__, "..", "data", "example_counts.csv")
        @test_throws ArgumentError fit_stan_with_pigeons(
            "/nonexistent/model.stan", csv_path)
        @test_throws ArgumentError fit_stan_with_pigeons(
            stan_model_path(:efdm), "/nonexistent/data.csv")
    end

end
